"""Canonical candidate route for the CUDA-native Full Evolution V2 owner.

The provider consumes one already-concretized artifact/device/B/K receipt.  It
does not build, tune, select another device, or silently repartition the batch.
Runtime authority is absolute ticks; generation-shaped coordinates are never
created by this route.  CRCA occurrence publication additionally requires the
Scheduler-owned attempt-authority receipt and never promotes a shadow identity.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import tempfile
from time import perf_counter, perf_counter_ns, process_time_ns
from typing import Any, Callable, Final, Mapping, Protocol, Sequence
from uuid import uuid4

import yaml
from . import cuda_full_evolution_v2_perf_surface as _cuda_v2_perf_surface
from emergence_contracts import (
    COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1,
    CanonicalPayloadV1,
    ContentDigest,
    SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
    SchemaVersion,
    ScienceObservationRecordV1,
    canonical_science_record_stream_sha256_v1,
    validate_science_native_control_receipt_v1,
    validate_science_observation_stream_v1,
)
from engine.native_runtime import (
    cuda_snapshot_publication_v1 as _native_snapshot_publication,
    cuda_snapshot_v1 as _native_snapshot_capsule,
    cuda_v2_compact_science_capture_v1 as _compact_science_capture,
    science_observation_projection_v1 as _science_observation_projection,
)
from engine.native_runtime.cuda_snapshot_encoder_v1.native_loader import (
    load_native_snapshot_encoder_v1,
)
from engine.native_runtime.cuda_snapshot_plan_v1 import (
    NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
    compile_cuda_snapshot_plan_v1,
)
from engine.native_runtime.cuda_snapshot_v1.default_binding import (
    resolve_snapshot_extension_path_v1,
)
from .canonical import canonical_bytes, canonical_sha256
from .cuda_full_evolution_v2_shared_route import (
    CudaFullEvolutionV2SharedRouteError,
    SHARED_W32_ALLOWED_TEMPORAL_K_V1,
    SHARED_W32_DEFAULT_LOGICAL_WORLDS_V1,
    SHARED_W32_GPU_POLICY_ID_V1,
    SHARED_W32_MAXIMUM_PHYSICAL_B_V1,
    SHARED_W32_MAX_ACS_SLOTS_V1,
    SHARED_W32_MINIMUM_PHYSICAL_B_V1,
    SHARED_W32_PEAK_VRAM_GIB_V1,
    validate_shared_w32_plan_v1,
)
from .acs_native_streamed_checkpoint_v1 import (
    DEFAULT_CHUNK_BYTES as ACS2_CAPTURE_CHUNK_BYTES_V1,
)
from .artifact_lifecycle import (
    ArtifactLifecycleError,
    PROGRESSIVE_STORAGE_RESERVATION_MODE_V1,
    QUALIFICATION,
    RunStorageGuard,
    StorageBudgetExceeded,
    build_sealed_artifact_inventory_row_v1,
    lifecycle_profile_contract_v1,
    progressive_storage_reservation_window_bytes_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from .contracts import (
    CampaignPlanV1,
    ExecutionRouteContractV1,
    MetricFamilyV1,
    ParameterContractV1,
    RuntimeProfileV1,
    UnsupportedReasonV1,
)
from .providers import (
    ACS_NATIVE_RUNTIME_ABI_V1,
    ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    CampaignProviderError,
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
    _acs_native_runtime_artifact_binding_v1,
    default_execution_provider_registry,
)
from .qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from .registry import ProfileRegistryV1


_cuda_v2_contract = _cuda_v2_perf_surface.contract_v2
_direct_sparse_runtime = _cuda_v2_perf_surface.direct_sparse_runtime_v2
_cuda_v2_layout = _cuda_v2_perf_surface.layout_v2
_cuda_v2_native_loader = _cuda_v2_perf_surface.native_loader_v2
_cuda_v2_runtime = _cuda_v2_perf_surface.runtime_v2
COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1 = (
    _compact_science_capture
    .COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1
)
COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1 = (
    _compact_science_capture.COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1
)
CompactScienceAsyncPublicationError = (
    _compact_science_capture.CompactScienceAsyncPublicationError
)
CompactScienceAsyncPublicationV1 = (
    _compact_science_capture.CompactScienceAsyncPublicationV1
)
COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1: Final[str] = (
    "EIEL_COMPACT_SCIENCE_SNAPSHOT_TIMING_V1"
)
COMPACT_SCIENCE_SNAPSHOT_TIMING_SUMMARY_SCHEMA_V1: Final[str] = (
    "eiel_compact_science_snapshot_timing_summary_v1"
)
COMPACT_SCIENCE_SNAPSHOT_IMPLEMENTATION_ENV_V1: Final[str] = (
    _native_snapshot_publication.NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1
)


CUDA_NATIVE_ABI_V1 = _cuda_v2_contract.CUDA_NATIVE_ABI_V1
CUDA_NATIVE_ROUTE_V2 = _cuda_v2_contract.CUDA_NATIVE_ROUTE_V2
CUDA_NATIVE_TIME_AUTHORITY_V1 = _cuda_v2_contract.CUDA_NATIVE_TIME_AUTHORITY_V1
FULL_EVOLUTION_CAPABILITIES_V2 = _cuda_v2_contract.FULL_EVOLUTION_CAPABILITIES_V2
TEMPORAL_K_MAX_V2 = _cuda_v2_contract.TEMPORAL_K_MAX_V2
TEMPORAL_K_MIN_V2 = _cuda_v2_contract.TEMPORAL_K_MIN_V2
CudaFullEvolutionNativeContractV2 = (
    _cuda_v2_contract.CudaFullEvolutionNativeContractV2
)
load_prebuilt_cuda_full_evolution_v2 = (
    _cuda_v2_native_loader.load_prebuilt_cuda_full_evolution_v2
)
validate_prebuilt_cuda_full_evolution_v2 = (
    _cuda_v2_native_loader.validate_prebuilt_cuda_full_evolution_v2
)
CHECKPOINT_SCHEMA_V2 = _cuda_v2_runtime.CHECKPOINT_SCHEMA_V2
EXECUTION_MODES_V2 = _cuda_v2_runtime.EXECUTION_MODES_V2
CudaFullEvolutionCheckpointV2 = _cuda_v2_runtime.CudaFullEvolutionCheckpointV2
CudaFullEvolutionRuntimeV2 = _cuda_v2_runtime.CudaFullEvolutionRuntimeV2
seal_event_sink_receipt_v2 = _cuda_v2_runtime.seal_event_sink_receipt_v2

PROFILE_ID: Final[str] = "eiel_full_evolution_cuda_native_v2"
PROVIDER_ID: Final[str] = "eiel-full-evolution-cuda-native-provider-v1"
WORLD_BASE: Final[str] = "tensor_world_cuda_full_evolution_v2"
CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1: Final[
    tuple[str, ...]
] = (
    "evolution.full_evolution",
    "runtime.production_cuda",
    "science.wrc_variability",
    "checkpoint.acs_two_slot",
    "authority.device_fallback",
)


def cuda_full_evolution_v2_wall_seconds_upper_bound_v1(total_ticks: int) -> int:
    """Return the provider's fail-closed absolute-tick wall-time bound."""

    if type(total_ticks) is not int or total_ticks <= 0:
        raise CampaignProviderError(
            "CUDA V2 total_ticks must be a positive integer for wall-time accounting"
        )
    return max(300, total_ticks * 60)


DEVICE_SELECTION_SCHEMA_V1: Final[str] = (
    "eiel_full_evolution_cuda_device_selection_receipt_v1"
)
DEVICE_SELECTION_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_device_selection_receipt_v2"
)
DENSE_EVENT_TRANSPORT_V2: Final[str] = "dense_v2"
DIRECT_SPARSE_EVENT_TRANSPORT_V3: Final[str] = "direct_sparse_v3"
EVENT_TRANSPORTS_V2: Final[tuple[str, str]] = (
    DENSE_EVENT_TRANSPORT_V2,
    DIRECT_SPARSE_EVENT_TRANSPORT_V3,
)
DIRECT_SPARSE_AUTOTUNE_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_autotune_v3"
)
DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_durable_envelope_v3"
)
DIRECT_SPARSE_MINIMUM_COMMITTED_WORLD_TICKS_PER_SECOND_V3: Final[float] = 446_719.311516
DIRECT_SPARSE_COMPILER_ACCOUNTING_SCHEMA_V1: Final[str] = (
    "eiel_signed_physical_checkpoint_planning_accounting_v1"
)
DIRECT_SPARSE_CHECKPOINT_ACCOUNTING_BASIS_V1: Final[str] = (
    "cuda_full_evolution_v2_signed_native_aggregate_v1"
)
DIRECT_SPARSE_FIXED_OUTPUT_BYTES_UPPER_BOUND_V1: Final[int] = 64 * 1024**2
ATTEMPT_AUTHORITY_NAME: Final[str] = "control/crca/attempt-authority.json"
CHECKPOINT_METADATA_SCHEMA_V1: Final[str] = "eiel_absolute_tick_checkpoint_metadata_v1"
CHECKPOINT_COMPLETION_SCHEMA_V1: Final[str] = (
    "eiel_absolute_tick_checkpoint_bundle_complete_v1"
)
PROVIDER_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_provider_checkpoint_v1"
)
EVENT_PERSISTENCE_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_event_persistence_v1"
)
EVENT_SINK_ID_V1: Final[str] = "cuda-full-evolution-v2-durable-run-scratch"
DIRECT_SPARSE_EVENT_SINK_ID_V3: Final[str] = (
    "cuda-full-evolution-v2-direct-sparse-durable-run-scratch-v3"
)
SCIENCE_CONTINUATION_ADMISSION_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_continuation_admission_v1"
)
SCIENCE_CONTRACT_RELATIVE_PATH_V1: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/"
    "science_consumer_extension_contract.json"
)
SCIENCE_OBSERVATION_OFF_V1: Final[str] = "OFF"
SCIENCE_OBSERVATION_TYPED_SIDECAR_V1: Final[str] = "TYPED_SIDECAR_V1"
SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1: Final[str] = "COMPACT_SIDECAR_V1"
SCIENCE_OBSERVATION_MODES_V1: Final[tuple[str, str, str]] = (
    SCIENCE_OBSERVATION_OFF_V1,
    SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
    SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1,
)
SCIENCE_CHECKPOINT_ADAPTER_CAPABILITY_V1: Final[str] = (
    "canonical_science_checkpoint_bundle_restore_adapter_v1"
)
SCIENCE_ROUTE_BINDING_TYPE_V1: Final[str] = "ScienceRouteBindingV1"
SCIENCE_OBSERVATION_PREFIX_CURSOR_TYPE_V1: Final[str] = (
    "ScienceObservationPrefixCursorV1"
)
CHECKPOINT_ARTIFACT_FAMILY_V1: Final[str] = "checkpoint.cuda_full_evolution_native_v2"
DEFAULT_FIXED_VRAM_RESERVE_BYTES_V1: Final[int] = 768 * 1024**2
MAX_PROFILE_BATCH_V1: Final[int] = 1_000_000
_ARTIFACT_NAMES: Final[tuple[str, str, str]] = (
    "cuda_extension",
    "extension_manifest",
    "device_selection_receipt",
)
HOST_EXECUTION_COMPONENT_IDENTITY_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_host_execution_component_identity_v1"
)
# These are the host-side authorities which can change execution semantics
# without changing the already-built CUDA image.  Keep this list explicit and
# ordered: it is a route contract input, not a replacement for the native
# manifest's complete historical source inventory.
HOST_EXECUTION_COMPONENT_PATHS_V1: Final[tuple[tuple[str, str], ...]] = (
    (
        "provider",
        "engine/campaigns/cuda_full_evolution_v2_provider.py",
    ),
    (
        "native_loader",
        "engine/perf/cuda_full_evolution_v2/native_loader.py",
    ),
    (
        "direct_sparse_runtime",
        "engine/perf/cuda_full_evolution_v2/direct_sparse_runtime.py",
    ),
    (
        "native_control",
        "engine/campaigns/science_cuda_v2_native_control_v1.py",
    ),
    (
        "science_capture",
        "engine/native_runtime/cuda_v2_science_capture_v1.py",
    ),
    (
        "compact_science_capture",
        "engine/native_runtime/cuda_v2_compact_science_capture_v1.py",
    ),
    (
        "compact_science_async_publication",
        "engine/native_runtime/compact_science_async_publication_v1.py",
    ),
    (
        "native_world_projector",
        "engine/campaigns/science_cuda_v2_native_world_v1.py",
    ),
    (
        "native_world_runtime",
        "engine/perf/cuda_full_evolution_v2/science_native_world_runtime_v1.py",
    ),
    (
        "science_projection",
        "engine/native_runtime/science_observation_projection_v1.py",
    ),
)
NATIVE_SNAPSHOT_HOST_COMPONENT_PATHS_V1: Final[
    tuple[tuple[str, str], ...]
] = (
    (
        "native_snapshot_plan",
        "engine/native_runtime/cuda_snapshot_plan_v1.py",
    ),
    (
        "native_snapshot_service",
        "engine/native_runtime/cuda_snapshot_service_v1.py",
    ),
    (
        "native_snapshot_publication",
        "engine/native_runtime/cuda_snapshot_publication_v1.py",
    ),
    (
        "native_snapshot_encoder_api",
        "engine/native_runtime/cuda_snapshot_encoder_v1/__init__.py",
    ),
    (
        "native_snapshot_encoder_loader",
        "engine/native_runtime/cuda_snapshot_encoder_v1/native_loader.py",
    ),
)

ArtifactValidatorV1 = Callable[[str | Path], tuple[Path, Path, dict[str, Any]]]
ArtifactLoaderV1 = Callable[..., Any]
RuntimeInitializerV1 = Callable[..., Any]
RuntimeRestorerV1 = Callable[..., Any]
DirectSparseOwnerInitializerV3 = Callable[..., Any]
DirectSparseOwnerRestorerV3 = Callable[..., Any]
NativeSnapshotPublicationFactoryV1 = Callable[..., Any]
NativeSnapshotEncoderLoaderV1 = Callable[..., Any]
NativeSnapshotImplementationSelectorV1 = Callable[[], str]


class AbsoluteTickCheckpointOccurrenceAdapterV1(Protocol):
    """Composition capability injected across the campaign dependency boundary.

    The adapter, not this execution provider, owns CRCA DTO construction,
    Scheduler attempt-authority validation, checkpoint tree-digest law, and
    occurrence visibility/restore validation.
    """

    def seal_bundle_v1(
        self,
        bundle: Path,
        *,
        checkpoint_id: str,
        absolute_next_tick: int,
        owner_id: str,
        owner_digest_sha256: str,
    ) -> Mapping[str, Any]: ...

    def materialize_occurrence_v1(
        self,
        bundle: Path,
        occurrence_root: Path,
        *,
        attempt_authority_path: Path,
        checkpoint_id: str,
        absolute_next_tick: int,
        owner_id: str,
        owner_digest_sha256: str,
        source_batch_id: str,
        parent_checkpoint_digest: Mapping[str, str] | None,
    ) -> Mapping[str, Any]: ...

    def restore_occurrence_v1(
        self,
        bundle: Path,
        occurrence_root: Path,
        *,
        attempt_authority_path: Path,
        source_batch_id: str,
        expected_absolute_next_tick: int,
        expected_occurrence_digest: Mapping[str, str],
    ) -> Mapping[str, Any]: ...

    def authorize_physical_batch_member_v1(
        self,
        *,
        attempt_authority_path: Path,
        source_batch_id: str,
        resident_request: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("CUDA Native V2 plan must be a mapping")
    return dict(plan)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignProviderError(f"{name} must be a lowercase SHA-256")
    return value


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CampaignProviderError(f"{name} must be a JSON object")
    return dict(value)


def _regular_unlinked_file(path: Path, *, name: str) -> Path:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise CampaignProviderError(f"{name} is missing: {path}") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise CampaignProviderError(f"{name} is not an unlinked regular file")
    return path.resolve(strict=True)


def host_execution_component_identity_v1() -> dict[str, Any]:
    """Return the content identity of the host execution authorities.

    The native loader intentionally validates a retained binary against only
    its native/ABI dependency closure.  This separate route component binds
    the host code which drives that binary, so a host-only change produces a
    new route identity while retaining the binary and its direct-sparse tuning
    receipt for reuse.  Semantic campaign inputs are deliberately absent from
    this projection; they remain in their own science identity contracts.
    """

    paths = tuple(path for _component_id, path in HOST_EXECUTION_COMPONENT_PATHS_V1)
    try:
        source_identity, source_rows = (
            _cuda_v2_native_loader.source_component_identity_v2(paths)
        )
    except (
        OSError,
        TypeError,
        ValueError,
        _cuda_v2_native_loader.CudaFullEvolutionNativeLoadError,
    ) as exc:
        raise CampaignProviderError(
            f"host execution component identity is unavailable: {exc}"
        ) from exc
    rows_by_path = {str(row["path"]): row for row in source_rows}
    component_rows = [
        {
            "component_id": component_id,
            "path": path,
            "sha256": str(rows_by_path[path]["sha256"]),
        }
        for component_id, path in HOST_EXECUTION_COMPONENT_PATHS_V1
    ]
    identity_body = {
        "schema_version": HOST_EXECUTION_COMPONENT_IDENTITY_SCHEMA_V1,
        "components": component_rows,
    }
    return {
        "component_id": "cuda_full_evolution_v2_host_execution",
        "schema_version": HOST_EXECUTION_COMPONENT_IDENTITY_SCHEMA_V1,
        "components": component_rows,
        "source_rows_sha256": source_identity,
        "identity_sha256": canonical_sha256(identity_body),
        "native_artifact_dependency_scope": "native_manifest_live_dependency_closure_only",
        "native_artifact_rebuild_required": False,
        "direct_sparse_autotune_reuse_allowed": True,
    }


def _artifact_row_v1(
    plan: Mapping[str, Any], name: str, *, kind: str
) -> tuple[Path, dict[str, Any]]:
    artifacts = plan.get("artifacts")
    raw = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(raw, Mapping):
        raise CampaignProviderError(f"CUDA Native V2 artifact is missing: {name}")
    row = dict(raw)
    if set(row) != {"path", "sha256", "kind", "abi"}:
        raise CampaignProviderError(f"CUDA Native V2 artifact row is not exact: {name}")
    path = _regular_unlinked_file(
        Path(str(row.get("path", ""))).expanduser(), name=f"artifact {name}"
    )
    if _sha256_file(path) != row.get("sha256"):
        raise CampaignProviderError(f"CUDA Native V2 artifact hash mismatch: {name}")
    if row.get("kind") != kind or row.get("abi") != CUDA_NATIVE_ABI_V1:
        raise CampaignProviderError(
            f"CUDA Native V2 artifact contract mismatch: {name}"
        )
    return path, row


def _receipt_core_v1(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def _device_selection_v1(
    plan: Mapping[str, Any],
    *,
    binary_sha256: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    path, row = _artifact_row_v1(
        plan, "device_selection_receipt", kind="device_selection_receipt"
    )
    receipt = _json_object(path, name="CUDA Native V2 device selection receipt")
    expected_keys = {
        "schema_version",
        "profile_id",
        "provider_id",
        "route_id",
        "native_abi",
        "artifact",
        "device_identity",
        "capacity_basis",
        "tuning_binding",
        "selection",
        "created_at_utc",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise CampaignProviderError("device selection receipt has an invalid shape")
    supplied = _sha256_text(
        receipt.get("receipt_sha256"), name="device selection receipt digest"
    )
    if supplied != canonical_sha256(_receipt_core_v1(receipt)):
        raise CampaignProviderError("device selection receipt digest mismatch")
    selection_schema = receipt.get("schema_version")
    if (
        selection_schema not in {DEVICE_SELECTION_SCHEMA_V1, DEVICE_SELECTION_SCHEMA_V2}
        or receipt.get("profile_id") != PROFILE_ID
        or receipt.get("provider_id") != PROVIDER_ID
        or receipt.get("route_id") != CUDA_NATIVE_ROUTE_V2
        or receipt.get("native_abi") != CUDA_NATIVE_ABI_V1
    ):
        raise CampaignProviderError("device selection receipt identity mismatch")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping) or artifact != {
        "binary_sha256": binary_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": manifest_sha256,
    }:
        raise CampaignProviderError("device selection binds another artifact unit")
    device = receipt.get("device_identity")
    if not isinstance(device, Mapping) or not device:
        raise CampaignProviderError("device selection lacks exact device identity")
    total_memory = device.get("total_memory_bytes")
    index = device.get("index")
    if type(total_memory) is not int or total_memory <= 0 or type(index) is not int:
        raise CampaignProviderError("device selection memory/index identity is invalid")
    capacity = receipt.get("capacity_basis")
    if (
        not isinstance(capacity, Mapping)
        or set(capacity)
        != {
            "total_vram_bytes",
            "fixed_reserve_bytes",
            "working_fraction_numerator",
            "working_fraction_denominator",
            "transient_free_vram_used",
        }
        or capacity.get("total_vram_bytes") != total_memory
        or capacity.get("working_fraction_numerator") != 3
        or capacity.get("working_fraction_denominator") != 4
        or capacity.get("transient_free_vram_used") is not False
        or type(capacity.get("fixed_reserve_bytes")) is not int
        or capacity["fixed_reserve_bytes"] < 0
    ):
        raise CampaignProviderError("device selection capacity basis is invalid")
    direct_sparse = selection_schema == DEVICE_SELECTION_SCHEMA_V2
    event_transport = (
        DIRECT_SPARSE_EVENT_TRANSPORT_V3 if direct_sparse else DENSE_EVENT_TRANSPORT_V2
    )
    tuning = receipt.get("tuning_binding")
    tuning_keys = {
        "path",
        "file_sha256",
        "schema_version",
        "temporal_k",
        "execution_mode",
    }
    if direct_sparse:
        tuning_keys.update(
            {
                "event_transport",
                "autotune_result_sha256",
                "selected_benchmark_body_sha256",
                "selected_benchmark_file_sha256",
            }
        )
    if (
        not isinstance(tuning, Mapping)
        or set(tuning) != tuning_keys
        or type(tuning.get("temporal_k")) is not int
        or not TEMPORAL_K_MIN_V2 <= tuning["temporal_k"] <= TEMPORAL_K_MAX_V2
        or tuning.get("execution_mode") not in EXECUTION_MODES_V2
        or (direct_sparse and tuning.get("event_transport") != event_transport)
    ):
        raise CampaignProviderError("device selection tuning binding is invalid")
    tuning_path = _regular_unlinked_file(
        Path(str(tuning["path"])).expanduser(),
        name="bound CUDA Native V2 tuning receipt",
    )
    if _sha256_file(tuning_path) != _sha256_text(
        tuning["file_sha256"], name="tuning receipt file digest"
    ):
        raise CampaignProviderError("bound tuning receipt file digest mismatch")
    tuned_k, tuning_receipt = _autotune_selection_v1(
        tuning_path,
        artifact_sha256=binary_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device,
        event_transport=event_transport,
    )
    if tuning_receipt.get("schema_version") != tuning.get(
        "schema_version"
    ) or tuned_k != tuning.get("temporal_k"):
        raise CampaignProviderError("device selection tuning source differs")
    if direct_sparse:
        normalized = tuning_receipt.get("_provider_selection")
        if (
            not isinstance(normalized, Mapping)
            or tuning.get("autotune_result_sha256")
            != normalized.get("autotune_result_sha256")
            or tuning.get("selected_benchmark_body_sha256")
            != normalized.get("result_body_sha256")
            or tuning.get("selected_benchmark_file_sha256")
            != normalized.get("result_file_sha256")
        ):
            raise CampaignProviderError(
                "device selection direct-sparse tuning source differs"
            )
    selection = receipt.get("selection")
    expected_selection_keys = {
        "requested_batch_size",
        "selected_batch_size",
        "temporal_k",
        "execution_mode",
        "required_bytes",
        "maximum_working_bytes",
        "headroom_bytes",
        "admitted",
    }
    if direct_sparse:
        expected_selection_keys.update(
            {
                "event_transport",
                "initial_capacity_rows_per_world",
                "horizon_dependent_resident_bytes",
            }
        )
    if (
        not isinstance(selection, Mapping)
        or set(selection) != expected_selection_keys
        or selection.get("admitted") is not True
        or type(selection.get("requested_batch_size")) is not int
        or type(selection.get("selected_batch_size")) is not int
        or selection["selected_batch_size"] <= 0
        or selection["requested_batch_size"] < selection["selected_batch_size"]
        or selection.get("temporal_k") != tuning.get("temporal_k")
        or selection.get("execution_mode") != tuning.get("execution_mode")
        or (
            direct_sparse
            and (
                selection.get("event_transport") != event_transport
                or selection.get("selected_batch_size") != normalized.get("batch_size")
                or selection.get("initial_capacity_rows_per_world")
                != normalized.get("initial_capacity_rows_per_world")
                or selection.get("required_bytes")
                != normalized.get("estimated_peak_bytes")
                or selection.get("horizon_dependent_resident_bytes") != 0
                or selection["requested_batch_size"] != selection["selected_batch_size"]
                or type(selection.get("required_bytes")) is not int
                or selection["required_bytes"] <= 0
                or type(selection.get("maximum_working_bytes")) is not int
                or selection["required_bytes"] > selection["maximum_working_bytes"]
                or selection.get("headroom_bytes")
                != total_memory
                - int(capacity["fixed_reserve_bytes"])
                - selection["required_bytes"]
            )
        )
    ):
        raise CampaignProviderError("device selection admission is invalid")
    return {
        "path": str(path),
        "file_sha256": row["sha256"],
        "receipt_sha256": supplied,
        "device_identity": dict(device),
        "capacity_basis": dict(capacity),
        "tuning_binding": dict(tuning),
        "selection": dict(selection),
        "event_transport": event_transport,
    }


def _science_continuation_contract_binding_v1(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = manifest.get("normative_execution_contracts")
    if not isinstance(rows, list):
        raise CampaignProviderError(
            "native manifest lacks normative science-continuation contracts"
        )
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("path") == SCIENCE_CONTRACT_RELATIVE_PATH_V1
    ]
    if len(matches) != 1 or set(matches[0]) != {"path", "sha256"}:
        raise CampaignProviderError(
            "native manifest lacks one exact science-continuation contract binding"
        )
    row = matches[0]
    contract_path = (
        Path(__file__).resolve().parents[2] / SCIENCE_CONTRACT_RELATIVE_PATH_V1
    )
    contract_path = _regular_unlinked_file(
        contract_path, name="science-continuation extension contract"
    )
    contract_sha256 = _sha256_text(
        row.get("sha256"), name="science-continuation extension contract digest"
    )
    if _sha256_file(contract_path) != contract_sha256:
        raise CampaignProviderError(
            "science-continuation extension contract differs from the native build"
        )
    # This provider deliberately does not interpret a science branch's private
    # wire fields.  The native artifact must bind the additive contract by
    # content, while the fast/benchmark route remains observation-OFF and every
    # base checkpoint remains non-science-continuable.  A future science route
    # must supply and validate its canonical bundle/restore adapter separately.
    _json_object(contract_path, name="science-continuation extension contract")
    return {
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "required_checkpoint_adapter_capability": (
            SCIENCE_CHECKPOINT_ADAPTER_CAPABILITY_V1
        ),
        "required_canonical_identity_types": [
            SCIENCE_ROUTE_BINDING_TYPE_V1,
            SCIENCE_OBSERVATION_PREFIX_CURSOR_TYPE_V1,
        ],
        "checkpoint_adapter_present": False,
        "checkpoint_adapter_validated": False,
        "science_route_qualified": False,
    }


def _base_checkpoint_science_admission_v1(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    binding = artifact.get("science_continuation_contract")
    if not isinstance(binding, Mapping):
        raise CampaignProviderError(
            "artifact lacks its science-continuation prerequisite"
        )
    return {
        "schema_version": SCIENCE_CONTINUATION_ADMISSION_SCHEMA_V1,
        "status": "BASE_CHECKPOINT_ONLY_NOT_SCIENCE_CONTINUABLE",
        "observation_mode": SCIENCE_OBSERVATION_OFF_V1,
        "contract_sha256": binding["contract_sha256"],
        "required_checkpoint_adapter_capability": binding[
            "required_checkpoint_adapter_capability"
        ],
        "required_canonical_identity_types": list(
            binding["required_canonical_identity_types"]
        ),
        "checkpoint_adapter_present": False,
        "checkpoint_adapter_validated": False,
        "science_route_qualified": False,
        "science_continuable": False,
    }


def _artifact_unit_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
    *,
    validator: ArtifactValidatorV1 = validate_prebuilt_cuda_full_evolution_v2,
) -> dict[str, Any]:
    value = _payload(plan)
    binary, binary_row = _artifact_row_v1(
        value, "cuda_extension", kind="cuda_extension"
    )
    manifest_path, manifest_row = _artifact_row_v1(
        value, "extension_manifest", kind="extension_manifest"
    )
    try:
        validated_binary, validated_manifest, manifest = validator(binary)
    except Exception as exc:
        raise CampaignProviderError(
            f"CUDA Native V2 artifact validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        validated_binary.resolve() != binary
        or validated_manifest.resolve() != manifest_path
    ):
        raise CampaignProviderError(
            "declared binary and companion manifest are not the validated artifact unit"
        )
    if manifest_path != binary.with_suffix(binary.suffix + ".manifest.json"):
        raise CampaignProviderError("declared companion manifest path is not exact")
    if (
        manifest.get("output", {}).get("sha256") != binary_row["sha256"]
        or manifest.get("native_abi") != CUDA_NATIVE_ABI_V1
        or manifest.get("route_id") != CUDA_NATIVE_ROUTE_V2
    ):
        raise CampaignProviderError("validated manifest differs from declared artifact")
    manifest_sha256 = _sha256_text(
        manifest.get("manifest_sha256"), name="native manifest identity"
    )
    science_continuation_contract = _science_continuation_contract_binding_v1(manifest)
    selection = _device_selection_v1(
        value,
        binary_sha256=binary_row["sha256"],
        manifest_file_sha256=manifest_row["sha256"],
        manifest_sha256=manifest_sha256,
    )
    return {
        "binary_path": str(binary),
        "binary_sha256": binary_row["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_row["sha256"],
        "manifest_sha256": manifest_sha256,
        "manifest": dict(manifest),
        "science_continuation_contract": science_continuation_contract,
        "selection": selection,
    }


def cuda_full_evolution_v2_profile_v1() -> RuntimeProfileV1:
    metrics = MetricFamilyV1(
        family_id="cuda_full_evolution_v2_capability_counters",
        metrics=(),
        required_state_components=FULL_EVOLUTION_CAPABILITIES_V2,
        null_semantics="provider_result_counters_are_technical_only",
    )
    return RuntimeProfileV1(
        profile_id=PROFILE_ID,
        profile_version=2,
        backend="cuda_resident",
        world_base=WORLD_BASE,
        world_overlays=frozenset(),
        policy_family="fixed_mlp",
        selection_modes=frozenset({"standard"}),
        observation_contract="cuda_native_full_evolution_state_v2",
        state_components=FULL_EVOLUTION_CAPABILITIES_V2,
        checkpoint_components=(
            "canonical_native_state",
            "absolute_tick_frontier",
            "rng_identity",
            "event_sink_frontier",
            "science_continuation_admission",
        ),
        parameter_contracts={
            "cuda_native.temporal_k": ParameterContractV1(
                "cuda_native.temporal_k",
                int,
                lambda value: TEMPORAL_K_MIN_V2 <= value <= TEMPORAL_K_MAX_V2,
                "sealed CUDA dispatch temporal K",
            ),
            "cuda_native.execution_mode": ParameterContractV1(
                "cuda_native.execution_mode",
                str,
                lambda value: value in EXECUTION_MODES_V2,
                "sealed eager or CUDA Graph execution mode",
            ),
            "cuda_native.event_transport": ParameterContractV1(
                "cuda_native.event_transport",
                str,
                lambda value: value in EVENT_TRANSPORTS_V2,
                "sealed dense V2 or direct-sparse V3 event transport",
            ),
            "cuda_native.science_observation_mode": ParameterContractV1(
                "cuda_native.science_observation_mode",
                str,
                lambda value: value in SCIENCE_OBSERVATION_MODES_V1,
                "explicit OFF or engineering typed-sidecar science acquisition mode",
            ),
        },
        uniform_batch_parameters=(
            "cuda_native.temporal_k",
            "cuda_native.execution_mode",
            "cuda_native.event_transport",
            "cuda_native.science_observation_mode",
        ),
        metric_families=(metrics,),
        execution_owner="cuda_full_evolution_v2_native_owner",
        arbitrary_positive_generations=True,
        continuation_supported=True,
        cancellation_supported=False,
        batch_min=1,
        batch_max=MAX_PROFILE_BATCH_V1,
        qualified_batch_min=1,
        qualified_batch_max=MAX_PROFILE_BATCH_V1,
        generation_counter_bits=64,
        qualification_invariants={
            "route_id": CUDA_NATIVE_ROUTE_V2,
            "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
            "transition_oracle": "cuda_golden_only",
            "world_width": 32,
            "initial_population": 256,
            "resident_capacity": 512,
            "formal_qualification_authority": "none_execution_does_not_self_grant",
        },
        qualification_status="experimental",
        limitations=(
            "Experimental candidate route; execution grants no Qualification or production authority.",
            "A Scheduler-owned CRCA attempt authority is required for canonical occurrence publication and continuation.",
            "The selected device, B, K, artifact, execution mode, and event transport are immutable inputs.",
            "Base checkpoints are explicitly not science-continuable without the separate exact science extension and qualification.",
        ),
    )


def cuda_full_evolution_v2_profile_registry_v1() -> ProfileRegistryV1:
    registry = ProfileRegistryV1()
    registry.register(cuda_full_evolution_v2_profile_v1())
    return registry


def cuda_full_evolution_v2_provider_registry_v1(
    *,
    checkpoint_occurrence_adapter: (
        AbsoluteTickCheckpointOccurrenceAdapterV1 | None
    ) = None,
) -> ExecutionProviderRegistryV1:
    registry = default_execution_provider_registry()
    registry.register(
        CudaFullEvolutionV2ExecutionProviderV1(
            checkpoint_occurrence_adapter=checkpoint_occurrence_adapter
        )
    )
    return registry


def is_cuda_full_evolution_v2_spec_v1(spec: Mapping[str, Any]) -> bool:
    world = spec.get("world")
    return isinstance(world, Mapping) and world.get("base") == WORLD_BASE


def _batch_and_cells_v1(
    plan: Mapping[str, Any], batch_id: str
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Mapping[str, Any]]]:
    batches = plan.get("physical_batches")
    batch = next(
        (
            dict(row)
            for row in batches or ()
            if isinstance(row, Mapping) and row.get("batch_id") == batch_id
        ),
        None,
    )
    if batch is None:
        raise CampaignProviderError(f"CUDA Native V2 batch is absent: {batch_id}")
    cell_ids = tuple(str(item) for item in batch.get("cell_ids", ()))
    cells = {
        str(row.get("cell_id")): row
        for row in plan.get("cells", ())
        if isinstance(row, Mapping)
    }
    if not cell_ids or any(cell_id not in cells for cell_id in cell_ids):
        raise CampaignProviderError("CUDA Native V2 batch cell binding is invalid")
    return batch, cell_ids, cells


def _uniform_value_v1(request: ResidentBatchRequestV1, name: str) -> Any:
    values = dict(request.parameter_arrays).get(name)
    if not isinstance(values, tuple) or not values or len(set(values)) != 1:
        raise CampaignProviderError(f"CUDA Native V2 parameter is not uniform: {name}")
    return values[0]


def _plan_science_observation_mode_v1(plan: Mapping[str, Any]) -> str:
    """Read the explicit plan-level acquisition mode for route projection."""

    materialization = plan.get("science_materialization")
    if isinstance(materialization, Mapping):
        candidate = materialization.get("observation_mode")
        if candidate is not None:
            if candidate not in SCIENCE_OBSERVATION_MODES_V1:
                raise CampaignProviderError(
                    "plan science observation mode is unsupported"
                )
            return str(candidate)
    modes: set[str] = set()
    for cell in plan.get("cells", ()):
        if not isinstance(cell, Mapping):
            continue
        parameters = cell.get("parameters")
        if isinstance(parameters, Mapping):
            candidate = parameters.get(
                "cuda_native.science_observation_mode", SCIENCE_OBSERVATION_OFF_V1
            )
            modes.add(str(candidate))
    if not modes:
        return SCIENCE_OBSERVATION_OFF_V1
    if len(modes) != 1 or next(iter(modes)) not in SCIENCE_OBSERVATION_MODES_V1:
        raise CampaignProviderError(
            "plan science observation mode is not uniform or unsupported"
        )
    return next(iter(modes))


def _science_native_world_controls_v1(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate the ordered generic native-world controls for one batch."""

    from .science_cuda_v2_native_world_v1 import (
        ScienceCudaV2NativeWorldError,
        ScienceCudaV2NativeWorldV1,
        native_world_control_set_sha256_v1,
        project_science_cuda_v2_native_world_v1,
    )

    cells = [
        row
        for row in plan.get("cells", ())
        if isinstance(row, Mapping) and "science_semantic_inputs" in row
    ]
    if not cells:
        return ()
    controls: list[ScienceCudaV2NativeWorldV1] = []
    for index, cell in enumerate(cells):
        cell_id = cell.get("cell_id", index)
        try:
            expected = project_science_cuda_v2_native_world_v1(
                {
                    "semantic_inputs": cell["science_semantic_inputs"],
                    "science": cell.get("science_ownership"),
                }
            )
            applied = ScienceCudaV2NativeWorldV1.from_dict(
                cell.get("native_world_control")
            )
        except ScienceCudaV2NativeWorldError as exc:
            raise CampaignProviderError(
                f"science cell {cell_id} native-world control is invalid: {exc}"
            ) from exc
        if applied != expected:
            raise CampaignProviderError(
                f"science cell {cell_id} native-world control differs from semantic projection"
            )
        controls.append(applied)
    set_digest = native_world_control_set_sha256_v1(controls)
    materialization = plan.get("science_materialization")
    expected_binding = {"algorithm": "sha256", "hex": set_digest}
    expected_digests = [value.control_digest for value in controls]
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("native_world_control_set_sha256") != expected_binding
        or materialization.get("ordered_native_world_control_digests")
        != expected_digests
    ):
        raise CampaignProviderError(
            "science materialization native-world control set is absent or stale"
        )
    physical_batches = [
        row for row in plan.get("physical_batches", ()) if isinstance(row, Mapping)
    ]
    execution = plan.get("execution")
    if (
        len(physical_batches) != 1
        or physical_batches[0].get("native_world_control_set_sha256")
        != expected_binding
        or not isinstance(execution, Mapping)
        or execution.get("native_world_control_set_sha256") != expected_binding
    ):
        raise CampaignProviderError(
            "native-world control set binding was lost at an execution boundary"
        )
    return tuple(value.to_dict() for value in controls)


def _science_snapshot_ticks_v1(
    plan: Mapping[str, Any],
    request: ResidentBatchRequestV1 | None = None,
) -> tuple[int, ...]:
    """Return the declared absolute-tick snapshot union for selected worlds."""

    selected_ids = None if request is None else set(request.cell_ids)
    ticks: set[int] = set()
    for cell in plan.get("cells", ()):
        if not isinstance(cell, Mapping):
            continue
        cell_id = cell.get("cell_id")
        if selected_ids is not None and cell_id not in selected_ids:
            continue
        acquisition = cell.get("science_acquisition")
        if not isinstance(acquisition, Mapping):
            continue
        values = acquisition.get("snapshot_ticks", ())
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise CampaignProviderError(
                f"science cell {cell_id} snapshot_ticks is not an integer array"
            )
        for value in values:
            if type(value) is not int or value < 0:
                raise CampaignProviderError(
                    f"science cell {cell_id} snapshot tick is invalid"
                )
            ticks.add(value)
    return tuple(sorted(ticks))


def _science_native_control_v1(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate and return one exact science native-control receipt.

    The carrier may leave unsupported legacy semantic controls unprojected so
    host-only planning remains inspectable.  This provider seam is the
    pre-launch authority: any materialized science semantic input must pass the
    closed projection, and every applied receipt must agree with it exactly.
    """

    from .science_cuda_v2_native_control_v1 import (
        ScienceCudaV2NativeControlError,
        ScienceCudaV2NativeControlV1,
        derive_science_cuda_v2_native_rng_identity_v1,
        project_science_cuda_v2_native_control_v1,
    )

    cells = [
        row
        for row in plan.get("cells", ())
        if isinstance(row, Mapping) and "science_semantic_inputs" in row
    ]
    materialization = plan.get("science_materialization")
    raw_materialized = (
        materialization.get("native_control")
        if isinstance(materialization, Mapping)
        else None
    )
    materialized_control: ScienceCudaV2NativeControlV1 | None = None
    if raw_materialized is not None:
        try:
            materialized_control = ScienceCudaV2NativeControlV1.from_dict(
                raw_materialized
            )
        except ScienceCudaV2NativeControlError as exc:
            raise CampaignProviderError(
                f"science native-control receipt is invalid: {exc}"
            ) from exc
    if not cells:
        if materialized_control is not None:
            raise CampaignProviderError(
                "native-control receipt lacks bound science semantic cells"
            )
        return None

    raw_cell_controls = [cell.get("native_control") for cell in cells]
    scalar_digests = {
        canonical_sha256(value)
        for value in raw_cell_controls
        if isinstance(value, Mapping)
    }
    if any(value is None for value in raw_cell_controls) or len(scalar_digests) != 1:
        if raw_materialized is not None:
            raise CampaignProviderError(
                "heterogeneous native worlds carry an invalid batch-level scalar control"
            )
        for owner_name, owner in (
            ("execution", plan.get("execution")),
            (
                "physical batch",
                next(
                    (
                        row
                        for row in plan.get("physical_batches", ())
                        if isinstance(row, Mapping)
                    ),
                    None,
                ),
            ),
        ):
            if isinstance(owner, Mapping) and owner.get("native_control") is not None:
                raise CampaignProviderError(
                    f"{owner_name} carries a scalar control for heterogeneous worlds"
                )
        return None

    expected: ScienceCudaV2NativeControlV1 | None = None
    for index, cell in enumerate(cells):
        try:
            candidate = project_science_cuda_v2_native_control_v1(
                cell["science_semantic_inputs"]
            )
        except ScienceCudaV2NativeControlError as exc:
            raise CampaignProviderError(
                f"science cell {cell.get('cell_id', index)} has unsupported native control: {exc}"
            ) from exc
        if expected is None:
            expected = candidate
        elif candidate != expected:
            raise CampaignProviderError(
                "science batch mixes differing native control projections"
            )
        raw_cell_control = cell.get("native_control")
        if raw_cell_control is None:
            raise CampaignProviderError(
                "science cell is missing its exact applied native-control receipt"
            )
        try:
            cell_control = ScienceCudaV2NativeControlV1.from_dict(raw_cell_control)
        except ScienceCudaV2NativeControlError as exc:
            raise CampaignProviderError(
                f"science cell native-control receipt is invalid: {exc}"
            ) from exc
        if cell_control != candidate:
            raise CampaignProviderError(
                "science cell native-control receipt differs from semantic projection"
            )
        try:
            expected_native_uid, expected_native_identity = (
                derive_science_cuda_v2_native_rng_identity_v1(
                    {
                        "semantic_inputs": cell["science_semantic_inputs"],
                        "science": cell.get("science_ownership"),
                    }
                )
            )
        except ScienceCudaV2NativeControlError as exc:
            raise CampaignProviderError(
                f"science cell native RNG identity is invalid: {exc}"
            ) from exc
        if cell.get("native_rng_world_uid") != expected_native_uid:
            raise CampaignProviderError(
                "science cell native RNG UID differs from semantic pairing identity"
            )
        raw_native_identity = cell.get("native_rng_identity_sha256")
        if raw_native_identity != {
            "algorithm": "sha256",
            "hex": expected_native_identity,
        }:
            raise CampaignProviderError(
                "science cell native RNG identity digest differs from semantic pairing identity"
            )
        seeds = cell.get("seeds")
        if not isinstance(seeds, Mapping) or seeds.get("task") != expected_native_uid:
            raise CampaignProviderError(
                "science cell task seed differs from native RNG pairing identity"
            )
    assert expected is not None
    if not isinstance(materialization, Mapping) or raw_materialized is None:
        raise CampaignProviderError(
            "science materialization lacks its exact applied native-control receipt"
        )
    if materialized_control is not None and materialized_control != expected:
        raise CampaignProviderError(
            "science materialization native-control receipt differs from cells"
        )
    # The lowered campaign plan repeats the receipt at the execution and
    # physical-batch boundaries so launch/admission projections cannot silently
    # drop it.  Validate every repeated copy against the semantic projection.
    for owner_name, owner in (
        ("execution", plan.get("execution")),
        (
            "physical batch",
            next(
                (
                    row
                    for row in plan.get("physical_batches", ())
                    if isinstance(row, Mapping)
                ),
                None,
            ),
        ),
    ):
        if not isinstance(owner, Mapping) or "native_control" not in owner:
            raise CampaignProviderError(
                f"{owner_name} lacks its exact applied native-control receipt"
            )
        raw_owner = owner.get("native_control")
        if raw_owner is None:
            raise CampaignProviderError(
                f"{owner_name} lacks its exact applied native-control receipt"
            )
        try:
            owner_control = ScienceCudaV2NativeControlV1.from_dict(raw_owner)
        except ScienceCudaV2NativeControlError as exc:
            raise CampaignProviderError(
                f"{owner_name} native-control receipt is invalid: {exc}"
            ) from exc
        if owner_control != expected:
            raise CampaignProviderError(
                f"{owner_name} native-control receipt differs from cells"
            )
    return expected.to_dict()


def _validate_native_control_runtime_receipt_v1(
    receipt: object,
    expected: Mapping[str, Any] | None,
) -> None:
    """Require the direct native owner to echo the exact applied receipt."""

    if expected is None:
        return
    if not isinstance(receipt, Mapping):
        raise CampaignProviderError(
            "native direct-sparse execution emitted no control receipt"
        )
    if receipt.get("native_control") != dict(expected):
        raise CampaignProviderError(
            "native direct-sparse execution control receipt differs from the bound plan"
        )
    if receipt.get("capability_mask") != expected.get("capability_mask"):
        raise CampaignProviderError(
            "native direct-sparse execution capability mask differs from the bound plan"
        )


def _validate_native_world_control_runtime_receipt_v1(
    receipt: object,
    *,
    expected_controls: Sequence[Mapping[str, Any]],
    expected_control_set_sha256: str | None,
) -> None:
    """Require the native owner to echo the generic ordered control-set binding."""

    if not expected_controls:
        return
    if not isinstance(receipt, Mapping):
        raise CampaignProviderError(
            "native direct-sparse execution emitted no world-control receipt"
        )
    binding = receipt.get("native_world_control")
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version")
        != "eiel_science_native_world_runtime_binding_v1"
        or binding.get("control_set_sha256") != expected_control_set_sha256
        or not isinstance(binding.get("founder_registry_sha256"), str)
        or len(str(binding.get("founder_registry_sha256"))) != 64
        or not isinstance(binding.get("founder_application"), Mapping)
    ):
        raise CampaignProviderError(
            "native direct-sparse world-control receipt differs from the bound plan"
        )


def _native_contract_v1(
    request: ResidentBatchRequestV1,
) -> CudaFullEvolutionNativeContractV2:
    if request.total_ticks is None:
        raise CampaignProviderError("CUDA Native V2 request lacks absolute total ticks")
    return CudaFullEvolutionNativeContractV2(
        B=len(request.cell_ids),
        total_ticks=request.total_ticks,
        start_next_tick=request.start_next_tick,
        temporal_k=int(_uniform_value_v1(request, "cuda_native.temporal_k")),
    )


def _direct_sparse_resource_estimate_v3(
    contract: CudaFullEvolutionNativeContractV2,
    *,
    checkpoint_boundaries: int,
    sealed_selection: Mapping[str, Any],
    device_selection: Mapping[str, Any],
    acs_checkpoint_slots: int = 0,
    compact_science_capture: bool = False,
    compact_snapshot_boundaries: int = 0,
    progressive_storage_guard: bool = False,
    compact_async_workers: int = COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1,
    compact_async_pending_slots: int = (
        COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1
    ),
) -> dict[str, Any]:
    """Return the one signed-B/K direct-sparse resource arithmetic."""

    if type(checkpoint_boundaries) is not int or checkpoint_boundaries < 1:
        raise CampaignProviderError("direct-sparse checkpoint count must be positive")
    if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
        raise CampaignProviderError(
            "direct-sparse ACS checkpoint slots must be zero, one, or two"
        )
    if (
        type(compact_science_capture) is not bool
        or type(progressive_storage_guard) is not bool
        or type(compact_snapshot_boundaries) is not int
        or compact_snapshot_boundaries < 0
        or type(compact_async_workers) is not int
        or compact_async_workers not in {1, 2}
        or type(compact_async_pending_slots) is not int
        or compact_async_pending_slots not in {1, 2}
        or compact_async_workers > compact_async_pending_slots
    ):
        raise CampaignProviderError("compact science resource inputs are invalid")
    if (
        sealed_selection.get("selected_batch_size") != contract.B
        or sealed_selection.get("temporal_k") != contract.temporal_k
        or sealed_selection.get("event_transport") != DIRECT_SPARSE_EVENT_TRANSPORT_V3
        or sealed_selection.get("horizon_dependent_resident_bytes") != 0
        or type(sealed_selection.get("initial_capacity_rows_per_world")) is not int
        or sealed_selection["initial_capacity_rows_per_world"] <= 0
    ):
        raise CampaignProviderError(
            "direct-sparse resource estimate differs from sealed B/K capacity"
        )
    dispatches = (contract.total_ticks + contract.temporal_k - 1) // contract.temporal_k
    checkpoint_bytes_per_world = (
        _cuda_v2_layout.state_layout_v2().canonical_checkpoint_bytes_per_world
    )
    checkpoint_payload_bytes_per_boundary = contract.B * checkpoint_bytes_per_world
    checkpoint_metadata_bytes_per_boundary = 128 * 1024
    checkpoint_bytes_per_boundary = (
        checkpoint_payload_bytes_per_boundary + checkpoint_metadata_bytes_per_boundary
    )
    checkpoint_bytes = checkpoint_boundaries * checkpoint_bytes_per_boundary
    event_rows_per_dispatch = (
        contract.B
        * int(sealed_selection["initial_capacity_rows_per_world"])
    )
    event_payload_bytes_per_dispatch = (
        event_rows_per_dispatch
        * _cuda_v2_layout.CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
    )
    event_stream_bytes = event_payload_bytes_per_dispatch * dispatches
    compact_metric_row_bytes = 21 * 8
    compact_birth_provenance_row_bytes = 132
    compact_stored_death_metric_row_bytes = 20 * 4 + 8
    compact_stored_birth_provenance_row_bytes = 64 + 4
    def gzip_upper_bound_v1(value: int) -> int:
        # zlib's documented compressBound terms plus a conservative gzip
        # wrapper/member allowance.  Each dispatch file is one independent
        # deterministic member, including the valid empty-member case.
        return value + (value >> 12) + (value >> 14) + (value >> 25) + 64

    compact_event_bytes_per_dispatch = (
        gzip_upper_bound_v1(event_payload_bytes_per_dispatch)
        if compact_science_capture
        else 0
    )
    compact_event_metric_bytes_per_dispatch = (
        gzip_upper_bound_v1(
            event_rows_per_dispatch * compact_stored_death_metric_row_bytes
        )
        if compact_science_capture
        else 0
    )
    compact_birth_provenance_bytes_per_dispatch = (
        gzip_upper_bound_v1(
            event_rows_per_dispatch
            * compact_stored_birth_provenance_row_bytes
        )
        if compact_science_capture
        else 0
    )
    compact_empty_member_bytes = (
        gzip_upper_bound_v1(0) if compact_science_capture else 0
    )
    compact_event_metric_bytes = (
        compact_event_metric_bytes_per_dispatch * dispatches
    )
    compact_checkpoint_frontier_bytes = (
        checkpoint_boundaries * contract.B * 512 * compact_metric_row_bytes
        if compact_science_capture
        else 0
    )
    compact_birth_provenance_bytes = (
        compact_birth_provenance_bytes_per_dispatch * dispatches
    )
    compact_event_auxiliary_bytes_per_dispatch = (
        max(
            compact_event_metric_bytes_per_dispatch + compact_empty_member_bytes,
            compact_birth_provenance_bytes_per_dispatch
            + compact_empty_member_bytes,
        )
        if compact_science_capture
        else 0
    )
    compact_event_auxiliary_bytes = (
        compact_event_auxiliary_bytes_per_dispatch * dispatches
    )
    compact_event_bytes = compact_event_bytes_per_dispatch * dispatches
    compact_dispatch_metadata_bytes_per_dispatch = (
        (
            _compact_science_capture
            .COMPACT_SCIENCE_LOGICAL_WORLD_FIXED_BYTES_UPPER_BOUND_V1
            + contract.B
            * _compact_science_capture
            .COMPACT_SCIENCE_LOGICAL_WORLD_BYTES_PER_WORLD_UPPER_BOUND_V1
        )
        if compact_science_capture
        else 0
    )
    compact_dispatch_metadata_bytes = (
        compact_dispatch_metadata_bytes_per_dispatch * dispatches
    )
    compact_snapshot_bytes = (
        compact_snapshot_boundaries * contract.B * 512 * 4096
        if compact_science_capture
        else 0
    )
    if not compact_science_capture:
        compact_async_pending_slots = 0
        compact_async_workers = 0
    compact_async_frozen_host_bytes_per_dispatch = (
        event_payload_bytes_per_dispatch
        + event_rows_per_dispatch * compact_metric_row_bytes
        + event_rows_per_dispatch * compact_birth_provenance_row_bytes
        + compact_dispatch_metadata_bytes_per_dispatch
        + (
            _compact_science_capture
            .COMPACT_SCIENCE_SOURCE_SINK_RECEIPT_SHA256_BYTES_V1
            if compact_science_capture
            else 0
        )
        if compact_science_capture
        else 0
    )
    compact_async_frozen_queue_host_bytes = (
        compact_async_pending_slots
        * compact_async_frozen_host_bytes_per_dispatch
    )
    # A worker may concurrently retain NumPy selection copies, packed
    # uncompressed rows, BytesIO/gzip outputs, and parsed manifest metadata.
    # Four frozen-capsule equivalents plus two complete stored-output sets is
    # deliberately conservative and separately admission-visible.
    compact_async_worker_peak_host_bytes = (
        4 * compact_async_frozen_host_bytes_per_dispatch
        + 2
        * (
            compact_event_bytes_per_dispatch
            + compact_event_metric_bytes_per_dispatch
            + compact_birth_provenance_bytes_per_dispatch
        )
    )
    compact_async_worker_host_bytes = (
        compact_async_workers * compact_async_worker_peak_host_bytes
    )
    compact_async_host_bytes = (
        compact_async_frozen_queue_host_bytes
        + compact_async_worker_host_bytes
    )
    compact_async_staging_bytes = (
        compact_async_pending_slots
        * (
            compact_event_bytes_per_dispatch
            + compact_event_auxiliary_bytes_per_dispatch
            + compact_dispatch_metadata_bytes_per_dispatch
        )
    )
    compact_vram_bytes = (
        contract.B
        * (
            2 * 512
            + 2 * int(sealed_selection["initial_capacity_rows_per_world"])
        )
        * compact_metric_row_bytes
        + contract.B
        * (
            512
            + 2 * int(sealed_selection["initial_capacity_rows_per_world"])
        )
        * compact_birth_provenance_row_bytes
        + contract.B * 22 * 8
        if compact_science_capture
        else 0
    )
    # Acknowledged raw envelopes remain exact R0 lifecycle objects until the
    # terminal compact prefix is sealed.  This removes destructive per-dispatch
    # cleanup and makes crash recovery/inventory fail closed.
    direct_sparse_event_scratch_peak_bytes = event_stream_bytes
    fixed_output_bytes = DIRECT_SPARSE_FIXED_OUTPUT_BYTES_UPPER_BOUND_V1
    result = {
        "resource_estimator_id": "cuda_full_evolution_v2_direct_sparse_sealed_bk_v3",
        "vram_bytes_upper_bound": (
            int(sealed_selection["required_bytes"]) + compact_vram_bytes
        ),
        "wall_seconds_upper_bound": cuda_full_evolution_v2_wall_seconds_upper_bound_v1(
            contract.total_ticks
        ),
        "disk_bytes_upper_bound": (
            direct_sparse_event_scratch_peak_bytes
            + compact_async_staging_bytes
            + compact_event_bytes
            + compact_event_auxiliary_bytes
            + compact_dispatch_metadata_bytes
            + compact_checkpoint_frontier_bytes
            + compact_snapshot_bytes
            + checkpoint_bytes
            + fixed_output_bytes
        ),
        "event_bytes_upper_bound": direct_sparse_event_scratch_peak_bytes,
        "direct_sparse_event_stream_bytes_upper_bound": event_stream_bytes,
        "direct_sparse_event_scratch_peak_bytes_upper_bound": (
            direct_sparse_event_scratch_peak_bytes
        ),
        "event_materialization_mode": (
            "async_compact_with_terminal_raw_r0_release_v1"
            if progressive_storage_guard and compact_science_capture
            else "durable_envelope_stream_v3"
        ),
        "compact_science_event_bytes_upper_bound": compact_event_bytes,
        "compact_science_uncompressed_event_bytes_upper_bound": (
            event_stream_bytes if compact_science_capture else 0
        ),
        "compact_science_event_metric_bytes_upper_bound": compact_event_metric_bytes,
        "compact_science_birth_provenance_bytes_upper_bound": (
            compact_birth_provenance_bytes
        ),
        "compact_science_event_auxiliary_bytes_upper_bound": (
            compact_event_auxiliary_bytes
        ),
        "compact_science_dispatch_metadata_bytes_upper_bound": (
            compact_dispatch_metadata_bytes
        ),
        "compact_science_dispatch_payload_storage_encoding": (
            COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1
            if compact_science_capture
            else None
        ),
        "compact_science_checkpoint_frontier_bytes_upper_bound": (
            compact_checkpoint_frontier_bytes
        ),
        "compact_science_snapshot_bytes_upper_bound": compact_snapshot_bytes,
        "compact_science_vram_bytes_upper_bound": compact_vram_bytes,
        "compact_science_async_workers": compact_async_workers,
        "compact_science_async_pending_slots": compact_async_pending_slots,
        "compact_science_async_host_bytes_per_dispatch_upper_bound": (
            compact_async_frozen_host_bytes_per_dispatch
        ),
        "compact_science_async_logical_world_metadata_bytes_per_dispatch_upper_bound": (
            compact_dispatch_metadata_bytes_per_dispatch
        ),
        "compact_science_async_frozen_queue_host_bytes_upper_bound": (
            compact_async_frozen_queue_host_bytes
        ),
        "compact_science_async_worker_peak_host_bytes_per_worker_upper_bound": (
            compact_async_worker_peak_host_bytes
        ),
        "compact_science_async_worker_host_bytes_upper_bound": (
            compact_async_worker_host_bytes
        ),
        "compact_science_async_host_bytes_upper_bound": (
            compact_async_host_bytes
        ),
        "compact_science_async_staging_bytes_upper_bound": (
            compact_async_staging_bytes
        ),
        "compact_science_async_worker_cuda_references": 0,
        "fixed_output_bytes_upper_bound": fixed_output_bytes,
        "checkpoint_bytes_per_trajectory_boundary": checkpoint_bytes_per_world,
        "checkpoint_physical_payload_bytes_per_boundary": (
            checkpoint_payload_bytes_per_boundary
        ),
        "checkpoint_physical_metadata_bytes_per_boundary": (
            checkpoint_metadata_bytes_per_boundary
        ),
        "checkpoint_bytes_per_boundary_upper_bound": checkpoint_bytes_per_boundary,
        "checkpoint_bytes_upper_bound": checkpoint_bytes,
        "checkpoint_physical_object_count_upper_bound": checkpoint_boundaries,
        "checkpoint_representation": "aggregate_physical_batch_v1",
        "checkpoint_accounting_basis": (DIRECT_SPARSE_CHECKPOINT_ACCOUNTING_BASIS_V1),
        "physical_batch_cells": contract.B,
        "sealed_total_vram_bytes": device_selection["capacity_basis"][
            "total_vram_bytes"
        ],
        "sealed_fixed_reserve_bytes": device_selection["capacity_basis"][
            "fixed_reserve_bytes"
        ],
        "event_transport": DIRECT_SPARSE_EVENT_TRANSPORT_V3,
        "horizon_dependent_resident_bytes": 0,
        "acs_checkpoint_slots": acs_checkpoint_slots,
        "checkpoint_capture_mode": (
            "direct_resident_cuda_rows_v1" if acs_checkpoint_slots else "synchronous_v1"
        ),
        "acs_capture_pool_count": 1 if acs_checkpoint_slots else 0,
        "acs_pinned_slab_count": 2 if acs_checkpoint_slots else 0,
        "acs_pinned_host_bytes_upper_bound": (
            2 * ACS2_CAPTURE_CHUNK_BYTES_V1 if acs_checkpoint_slots else 0
        ),
        "acs_pending_capture_files_upper_bound": acs_checkpoint_slots,
        "acs_pending_capture_file_bytes_upper_bound": (
            acs_checkpoint_slots * checkpoint_payload_bytes_per_boundary
        ),
        "acs_producer_full_state_host_bytes": 0,
        "acs_producer_compacted_cuda_bytes": 0,
        "acs_worker_full_state_bytes": 0,
        "acs_worker_cuda_references": 0,
    }
    if progressive_storage_guard:
        if not compact_science_capture or acs_checkpoint_slots != 0:
            raise CampaignProviderError(
                "progressive storage guarding requires compact science capture "
                "without detached ACS slots"
            )
        dispatch_increment = (
            direct_sparse_event_scratch_peak_bytes
            + compact_async_staging_bytes
            + compact_event_bytes_per_dispatch
            + compact_event_auxiliary_bytes_per_dispatch
            + compact_dispatch_metadata_bytes_per_dispatch
        )
        compact_frontier_bytes_per_boundary = (
            contract.B * 512 * compact_metric_row_bytes
        )
        snapshot_bytes_per_boundary = (
            contract.B * 512 * 4096
            if compact_snapshot_boundaries
            else 0
        )
        boundary_increment = (
            checkpoint_bytes_per_boundary
            + compact_frontier_bytes_per_boundary
            + snapshot_bytes_per_boundary
        )
        emergency_checkpoint_increment = (
            checkpoint_bytes_per_boundary
            + compact_frontier_bytes_per_boundary
        )
        control_allowance = fixed_output_bytes
        window = (
            control_allowance
            + dispatch_increment
            + max(boundary_increment, emergency_checkpoint_increment)
        )
        result.update(
            {
                "storage_reservation_mode": (
                    PROGRESSIVE_STORAGE_RESERVATION_MODE_V1
                ),
                "storage_reservation_guard_scope": (
                    "provider_dispatch_and_checkpoint_boundary_v1"
                ),
                "storage_reservation_window_bytes_upper_bound": window,
                "storage_reservation_control_bytes_upper_bound": (
                    control_allowance
                ),
                "storage_reservation_dispatch_increment_bytes_upper_bound": (
                    dispatch_increment
                ),
                "storage_reservation_boundary_increment_bytes_upper_bound": (
                    boundary_increment
                ),
                "storage_reservation_emergency_checkpoint_bytes_upper_bound": (
                    emergency_checkpoint_increment
                ),
                "storage_reservation_total_envelope_bytes_upper_bound": result[
                    "disk_bytes_upper_bound"
                ],
            }
        )
        progressive_storage_reservation_window_bytes_v1(result)
    return result


def _native_snapshot_resource_accounting_v1(
    contract: CudaFullEvolutionNativeContractV2,
    *,
    slot_count: int,
    implementation: str = "native_v1",
) -> dict[str, Any]:
    """Return the exact horizon-independent preferred-transport reservation."""

    if type(slot_count) is not int or slot_count not in {1, 2}:
        raise CampaignProviderError(
            "native snapshot resource accounting requires one or two slots"
        )
    if implementation not in _native_snapshot_publication.NATIVE_SNAPSHOT_NATIVE_IMPLEMENTATIONS_V1:
        raise CampaignProviderError(
            "native snapshot resource accounting implementation is unsupported"
        )
    layout = _cuda_v2_layout.state_layout_v2()
    capacity = int(layout.field("agent_id").shape[0])
    metric_width = len(
        _compact_science_capture.COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    )
    resident_state_bytes_per_world = int(layout.state_bytes_per_world)
    capture_state_bytes_per_world = int(
        layout.canonical_checkpoint_bytes_per_world
    )
    state_bytes_per_slot = contract.B * capture_state_bytes_per_world
    metrics_bytes_per_slot = contract.B * capacity * metric_width * 8
    bytes_per_slot = state_bytes_per_slot + metrics_bytes_per_slot
    return {
        "native_snapshot_implementation": implementation,
        "native_snapshot_capture_mode": (
            NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
        ),
        "native_snapshot_slot_count": slot_count,
        "native_snapshot_resident_state_bytes_per_world": (
            resident_state_bytes_per_world
        ),
        "native_snapshot_capture_state_bytes_per_world": (
            capture_state_bytes_per_world
        ),
        "native_snapshot_resident_state_bytes_per_slot": (
            contract.B * resident_state_bytes_per_world
        ),
        "native_snapshot_state_bytes_per_slot": state_bytes_per_slot,
        "native_snapshot_metrics_bytes_per_slot": metrics_bytes_per_slot,
        "native_snapshot_bytes_per_slot": bytes_per_slot,
        "native_snapshot_pinned_host_bytes_upper_bound": (
            slot_count * bytes_per_slot
        ),
        "native_snapshot_queue_capacity": slot_count,
        "native_snapshot_worker_count": 1,
        "native_snapshot_horizon_dependent_buffer_bytes": 0,
    }


def _validate_native_snapshot_resource_admission_v1(
    plan: Mapping[str, Any],
    contract: CudaFullEvolutionNativeContractV2,
    *,
    implementation: str,
    slot_count: int,
) -> None:
    """Require execution mode to equal the plan-bound pinned-memory admission."""

    estimate = plan.get("resource_estimate")
    if not isinstance(estimate, Mapping):
        raise CampaignProviderError(
            "native snapshot execution lacks plan-bound resource accounting"
        )
    present = {
        str(name): value
        for name, value in estimate.items()
        if str(name).startswith("native_snapshot_")
    }
    if implementation == "python_v1":
        if present:
            raise CampaignProviderError(
                "python snapshot execution differs from the plan-bound native "
                "snapshot resource admission"
            )
        return
    expected = _native_snapshot_resource_accounting_v1(
        contract,
        slot_count=slot_count,
        implementation=implementation,
    )
    if (
        not _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
            implementation
        )
        or present != expected
    ):
        raise CampaignProviderError(
            "native snapshot execution differs from its plan-bound slot and "
            "pinned-memory resource admission"
        )


def _aggregate_shared_w32_resource_estimate_v1(
    estimate: Mapping[str, Any],
    *,
    physical_batch_count: int,
    logical_world_count: int,
) -> dict[str, Any]:
    """Aggregate serial output cost while retaining one-batch peak residency."""

    if (
        type(physical_batch_count) is not int
        or physical_batch_count < 1
        or type(logical_world_count) is not int
        or logical_world_count < 1
    ):
        raise CampaignProviderError("shared W32 resource aggregation is invalid")
    value = dict(estimate)
    for field in (
        "disk_bytes_upper_bound",
        "event_bytes_upper_bound",
        "compact_science_event_bytes_upper_bound",
        "compact_science_event_metric_bytes_upper_bound",
        "compact_science_birth_provenance_bytes_upper_bound",
        "compact_science_event_auxiliary_bytes_upper_bound",
        "compact_science_checkpoint_frontier_bytes_upper_bound",
        "compact_science_snapshot_bytes_upper_bound",
        "fixed_output_bytes_upper_bound",
        "checkpoint_bytes_upper_bound",
        "checkpoint_physical_object_count_upper_bound",
        "wall_seconds_upper_bound",
    ):
        raw = value.get(field)
        if type(raw) is int:
            value[field] = raw * physical_batch_count
    value.update(
        {
            "gpu_resource_policy_id": SHARED_W32_GPU_POLICY_ID_V1,
            "compiled_batch_count": physical_batch_count,
            "logical_world_count": logical_world_count,
            "serial_physical_batch_execution": True,
            "vram_peak_scope": "one_active_uniform_physical_batch",
            "disk_and_wall_scope": "all_serial_physical_batches",
        }
    )
    return value


def cuda_full_evolution_v2_signed_planning_resource_accounting_v1(
    *,
    spec: Mapping[str, Any],
    logical_cells: int,
    physical_batch_cells: int,
    compiled_batch_count: int,
    comparison_branch_count: int,
    artifact_validator: ArtifactValidatorV1 | None = None,
) -> dict[str, Any] | None:
    """Bind physical checkpoint accounting to the sealed artifact/device route.

    Dense V1 and every generic campaign retain compiler-conservative accounting.
    A direct-sparse request can reach this result only after the same artifact,
    manifest, tuning, device, B, and K validation used by provider admission.
    """

    if not is_cuda_full_evolution_v2_spec_v1(spec):
        raise CampaignProviderError(
            "signed CUDA Native V2 planning accounting requires its exact world"
        )
    parameters = spec.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CampaignProviderError(
            "signed CUDA Native V2 planning accounting lacks campaign parameters"
        )
    if (
        parameters.get("cuda_native.event_transport")
        != DIRECT_SPARSE_EVENT_TRANSPORT_V3
    ):
        return None
    artifact = _artifact_unit_v1(
        spec,
        validator=(
            validate_prebuilt_cuda_full_evolution_v2
            if artifact_validator is None
            else artifact_validator
        ),
    )
    selected = artifact["selection"]
    if selected["event_transport"] != DIRECT_SPARSE_EVENT_TRANSPORT_V3:
        return None
    selection = selected["selection"]
    execution = spec.get("execution")
    checkpoints = spec.get("checkpoints")
    if not all(
        isinstance(value, Mapping) for value in (execution, parameters, checkpoints)
    ):
        raise CampaignProviderError(
            "direct-sparse planning accounting lacks campaign route fields"
        )
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks") or ())
    temporal_k = parameters.get("cuda_native.temporal_k")
    event_transport = parameters.get("cuda_native.event_transport")
    execution_mode = parameters.get("cuda_native.execution_mode")
    runtime_manifest = artifact["manifest"].get("runtime")
    acs_checkpoint_slots = checkpoints.get("acs_slots", 0)
    shared_policy = (
        execution.get("gpu_resource_policy") == SHARED_W32_GPU_POLICY_ID_V1
    )
    physical_shape_valid = (
        (
            SHARED_W32_MINIMUM_PHYSICAL_B_V1
            <= physical_batch_cells
            <= SHARED_W32_MAXIMUM_PHYSICAL_B_V1
            and logical_cells == physical_batch_cells * compiled_batch_count
            and compiled_batch_count >= 1
            and temporal_k in SHARED_W32_ALLOWED_TEMPORAL_K_V1
            and type(acs_checkpoint_slots) is int
            and 0 <= acs_checkpoint_slots <= SHARED_W32_MAX_ACS_SLOTS_V1
        )
        if shared_policy
        else (
            logical_cells == physical_batch_cells
            and compiled_batch_count == 1
            and type(acs_checkpoint_slots) is int
            and acs_checkpoint_slots in {0, 1, 2}
        )
    )
    if (
        execution.get("execution_route_identity") != "tick_native_execution_route_v1"
        or execution.get("batch") != {"mode": "explicit", "cells": physical_batch_cells}
        or not physical_shape_valid
        or comparison_branch_count != 0
        or selection.get("selected_batch_size") != physical_batch_cells
        or selection.get("temporal_k") != temporal_k
        or selected["event_transport"] != event_transport
        or selection.get("execution_mode") != execution_mode
        or parameters.get("cuda_native.science_observation_mode")
        not in SCIENCE_OBSERVATION_MODES_V1
        or checkpoints.get("native_streamed_checkpoint") is not True
        or checkpoints.get("storage_mode") != "disk"
        or not checkpoint_ticks
        or not isinstance(runtime_manifest, Mapping)
        or runtime_manifest.get("production_owner")
        != "CudaFullEvolutionDirectSparseOwnerV3"
        or runtime_manifest.get("horizon_dependent_resident_bytes") != 0
    ):
        raise CampaignProviderError(
            "direct-sparse planning accounting route identity mismatch"
        )
    contract = CudaFullEvolutionNativeContractV2(
        B=physical_batch_cells,
        total_ticks=execution.get("total_ticks"),
        start_next_tick=execution.get("start_next_tick", 0),
        temporal_k=temporal_k,
    )
    estimate = _direct_sparse_resource_estimate_v3(
        contract,
        checkpoint_boundaries=len(checkpoint_ticks),
        sealed_selection=selection,
        device_selection=selected,
        acs_checkpoint_slots=acs_checkpoint_slots,
        compact_science_capture=(
            parameters.get("cuda_native.science_observation_mode")
            == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
        ),
        compact_snapshot_boundaries=len(_science_snapshot_ticks_v1(spec)),
    )
    if shared_policy:
        estimate = _aggregate_shared_w32_resource_estimate_v1(
            estimate,
            physical_batch_count=compiled_batch_count,
            logical_world_count=logical_cells,
        )
        if int(estimate["vram_bytes_upper_bound"]) > int(
            SHARED_W32_PEAK_VRAM_GIB_V1 * 1024**3
        ):
            raise CampaignProviderError(
                "shared W32 signed provider VRAM bound exceeds 1.25 GiB"
            )
    schedule_sha256 = canonical_sha256(
        {
            "time_authority": "absolute_tick_v1",
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    )
    device_identity_sha256 = canonical_sha256(selected["device_identity"])
    binding = {
        "schema_version": DIRECT_SPARSE_COMPILER_ACCOUNTING_SCHEMA_V1,
        "profile_id": PROFILE_ID,
        "artifact_binary_sha256": artifact["binary_sha256"],
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "device_selection_receipt_sha256": selected["receipt_sha256"],
        "device_identity_sha256": device_identity_sha256,
        "event_transport": DIRECT_SPARSE_EVENT_TRANSPORT_V3,
        "logical_cells": logical_cells,
        "physical_batch_cells": physical_batch_cells,
        **(
            {
                "compiled_batch_count": compiled_batch_count,
                "gpu_resource_policy_id": SHARED_W32_GPU_POLICY_ID_V1,
            }
            if shared_policy
            else {}
        ),
        "temporal_k": temporal_k,
        "total_ticks": contract.total_ticks,
        "checkpoint_schedule_sha256": schedule_sha256,
    }
    return {
        "schema_version": DIRECT_SPARSE_COMPILER_ACCOUNTING_SCHEMA_V1,
        "authority": "signed_provider_artifact_device_selection_v1",
        "profile_id": PROFILE_ID,
        "execution_route_identity": "tick_native_execution_route_v1",
        "resource_estimator_id": estimate["resource_estimator_id"],
        "resource_accounting_binding_sha256": canonical_sha256(binding),
        "artifact_binary_sha256": artifact["binary_sha256"],
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "device_selection_receipt_sha256": selected["receipt_sha256"],
        "device_identity_sha256": device_identity_sha256,
        "event_transport": DIRECT_SPARSE_EVENT_TRANSPORT_V3,
        "logical_cells": logical_cells,
        "physical_batch_cells": physical_batch_cells,
        **(
            {
                "compiled_batch_count": compiled_batch_count,
                "gpu_resource_policy_id": SHARED_W32_GPU_POLICY_ID_V1,
            }
            if shared_policy
            else {}
        ),
        "temporal_k": temporal_k,
        "total_ticks": contract.total_ticks,
        "checkpoint_schedule_sha256": schedule_sha256,
        "checkpoint_boundaries": len(checkpoint_ticks),
        "checkpoint_representation": estimate["checkpoint_representation"],
        "checkpoint_accounting_basis": estimate["checkpoint_accounting_basis"],
        "checkpoint_bytes_per_trajectory_boundary": estimate[
            "checkpoint_bytes_per_trajectory_boundary"
        ],
        "checkpoint_physical_payload_bytes_per_boundary": estimate[
            "checkpoint_physical_payload_bytes_per_boundary"
        ],
        "checkpoint_physical_metadata_bytes_per_boundary": estimate[
            "checkpoint_physical_metadata_bytes_per_boundary"
        ],
        "checkpoint_bytes_per_boundary_upper_bound": estimate[
            "checkpoint_bytes_per_boundary_upper_bound"
        ],
        "checkpoint_bytes_upper_bound": estimate["checkpoint_bytes_upper_bound"],
        "checkpoint_physical_object_count_upper_bound": estimate[
            "checkpoint_physical_object_count_upper_bound"
        ],
        "event_bytes_upper_bound": estimate["event_bytes_upper_bound"],
        "durable_noncheckpoint_bytes_upper_bound": (
            estimate["disk_bytes_upper_bound"]
            - estimate["checkpoint_bytes_upper_bound"]
            - estimate["event_bytes_upper_bound"]
            - estimate["fixed_output_bytes_upper_bound"]
        ),
        "fixed_output_bytes_upper_bound": estimate["fixed_output_bytes_upper_bound"],
        "disk_bytes_upper_bound": estimate["disk_bytes_upper_bound"],
        **(
            {
                "vram_bytes_upper_bound": estimate["vram_bytes_upper_bound"],
                "wall_seconds_upper_bound": estimate["wall_seconds_upper_bound"],
                "serial_physical_batch_execution": True,
            }
            if shared_policy
            else {}
        ),
        "horizon_dependent_resident_bytes": 0,
    }


def _initialize_direct_sparse_owner_v3(*args: Any, **kwargs: Any) -> Any:
    return _direct_sparse_runtime.CudaFullEvolutionDirectSparseOwnerV3.initialize(
        *args, **kwargs
    )


def _restore_direct_sparse_owner_v3(*args: Any, **kwargs: Any) -> Any:
    return _direct_sparse_runtime.CudaFullEvolutionDirectSparseOwnerV3.from_checkpoint(
        *args, **kwargs
    )


class CudaFullEvolutionV2ExecutionProviderV1:
    """Run one exact sealed CUDA Native V2 physical batch."""

    provider_id = PROVIDER_ID
    profile_id = PROFILE_ID
    resident_segment_generation_min = 1
    reporting_parity_supported = False

    @staticmethod
    def campaign_metric_rows(
        rows: Sequence[Mapping[str, Any]], *, cell_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Cover the generic cell index without inventing science metrics."""

        if rows:
            return [dict(row) for row in rows]
        return [{"cell_id": str(cell_id), "metrics": {}} for cell_id in cell_ids]

    def __init__(
        self,
        *,
        artifact_validator: ArtifactValidatorV1 = validate_prebuilt_cuda_full_evolution_v2,
        artifact_loader: ArtifactLoaderV1 = load_prebuilt_cuda_full_evolution_v2,
        runtime_initializer: RuntimeInitializerV1 = CudaFullEvolutionRuntimeV2.initialize,
        runtime_restorer: RuntimeRestorerV1 = CudaFullEvolutionRuntimeV2.from_checkpoint,
        sink_receipt_factory: Callable[..., Any] = seal_event_sink_receipt_v2,
        direct_sparse_owner_initializer: DirectSparseOwnerInitializerV3 = (
            _initialize_direct_sparse_owner_v3
        ),
        direct_sparse_owner_restorer: DirectSparseOwnerRestorerV3 = (
            _restore_direct_sparse_owner_v3
        ),
        checkpoint_occurrence_adapter: (
            AbsoluteTickCheckpointOccurrenceAdapterV1 | None
        ) = None,
        acs_require_native_pool: bool = True,
        acs_require_direct_resident_capture: bool = True,
        compact_science_async_workers: int = (
            COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1
        ),
        compact_science_async_pending_slots: int = (
            COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1
        ),
        native_snapshot_implementation_selector: (
            NativeSnapshotImplementationSelectorV1
        ) = _native_snapshot_publication.native_snapshot_implementation_v1,
        native_snapshot_encoder_loader: NativeSnapshotEncoderLoaderV1 = (
            load_native_snapshot_encoder_v1
        ),
        native_snapshot_publication_factory: (
            NativeSnapshotPublicationFactoryV1
        ) = _native_snapshot_publication.bind_native_snapshot_publication_service_v1,
        native_snapshot_slots: int = 2,
    ) -> None:
        self._artifact_validator = artifact_validator
        self._artifact_loader = artifact_loader
        self._runtime_initializer = runtime_initializer
        self._runtime_restorer = runtime_restorer
        self._sink_receipt_factory = sink_receipt_factory
        self._direct_sparse_owner_initializer = direct_sparse_owner_initializer
        self._direct_sparse_owner_restorer = direct_sparse_owner_restorer
        self._checkpoint_occurrence_adapter = checkpoint_occurrence_adapter
        if type(acs_require_native_pool) is not bool:
            raise CampaignProviderError(
                "CUDA Native V2 ACS native-pool policy must be an exact bool"
            )
        self._acs_require_native_pool = acs_require_native_pool
        if type(acs_require_direct_resident_capture) is not bool:
            raise CampaignProviderError(
                "CUDA Native V2 ACS direct-capture policy must be an exact bool"
            )
        self._acs_require_direct_resident_capture = (
            acs_require_direct_resident_capture
        )
        if (
            type(compact_science_async_workers) is not int
            or compact_science_async_workers not in {1, 2}
            or type(compact_science_async_pending_slots) is not int
            or compact_science_async_pending_slots not in {1, 2}
            or compact_science_async_workers
            > compact_science_async_pending_slots
        ):
            raise CampaignProviderError(
                "CUDA Native V2 compact-science async worker/slot policy is invalid"
            )
        self._compact_science_async_workers = compact_science_async_workers
        self._compact_science_async_pending_slots = (
            compact_science_async_pending_slots
        )
        if (
            not callable(native_snapshot_implementation_selector)
            or not callable(native_snapshot_encoder_loader)
            or not callable(native_snapshot_publication_factory)
        ):
            raise CampaignProviderError(
                "CUDA Native V2 native-snapshot integration hooks must be callable"
            )
        if type(native_snapshot_slots) is not int or native_snapshot_slots not in {
            1,
            2,
        }:
            raise CampaignProviderError(
                "CUDA Native V2 native-snapshot service supports one or two slots"
            )
        self._native_snapshot_implementation_selector = (
            native_snapshot_implementation_selector
        )
        self._native_snapshot_encoder_loader = native_snapshot_encoder_loader
        self._native_snapshot_publication_factory = (
            native_snapshot_publication_factory
        )
        self._native_snapshot_slots = native_snapshot_slots

    def bound_physical_batch_set_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        source_batch_id: str,
    ) -> dict[str, Any] | None:
        """Bind every uniform physical batch when shared W32 is selected."""

        value = _payload(plan)
        try:
            policy = validate_shared_w32_plan_v1(value)
        except CudaFullEvolutionV2SharedRouteError as exc:
            raise CampaignProviderError(str(exc)) from exc
        if policy is None:
            return None
        batches = value.get("physical_batches")
        assert isinstance(batches, list)
        rows = [dict(row) for row in batches if isinstance(row, Mapping)]
        batch_ids = [str(row.get("batch_id")) for row in rows]
        if len(rows) != len(batches) or source_batch_id not in batch_ids:
            raise CampaignProviderError(
                "shared W32 bound request is outside its physical batch set"
            )
        batch_cells = len(rows[0]["cell_ids"])
        partition = [
            {
                "ordinal": int(row["ordinal"]),
                "batch_id": str(row["batch_id"]),
                "cell_ids": list(row["cell_ids"]),
            }
            for row in rows
        ]
        request_bindings = [
            {
                "batch_id": batch_id,
                "resident_request_sha256": canonical_sha256(
                    self.translate_batch(value, batch_id).payload()
                ),
            }
            for batch_id in batch_ids
        ]
        return {
            "schema_version": (
                "eiel_cuda_full_evolution_v2_physical_batch_set_v1"
            ),
            "gpu_resource_policy_id": SHARED_W32_GPU_POLICY_ID_V1,
            "batch_ids": batch_ids,
            "physical_batch_count": len(rows),
            "uniform_physical_batch_cells": batch_cells,
            "logical_world_count": sum(len(row["cell_ids"]) for row in rows),
            "physical_partition_sha256": canonical_sha256(partition),
            "ordered_resident_request_bindings": request_bindings,
            "ordered_resident_request_bindings_sha256": canonical_sha256(
                request_bindings
            ),
        }

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        value = _payload(plan)
        _batch, cell_ids, cells = _batch_and_cells_v1(value, batch_id)
        execution = value.get("execution")
        if not isinstance(execution, Mapping):
            raise CampaignProviderError("CUDA Native V2 plan execution is missing")
        parameter_names = sorted(
            {
                str(name)
                for cell_id in cell_ids
                for name in cells[cell_id].get("parameters", {})
            }
        )
        seed_names = sorted(
            {
                str(name)
                for cell_id in cell_ids
                for name in cells[cell_id].get("seeds", {})
            }
        )
        control = execution.get("tick_native_control")
        if not isinstance(control, Mapping):
            raise CampaignProviderError("CUDA Native V2 tick-native control is missing")
        acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
            raise CampaignProviderError(
                "CUDA Native V2 ACS checkpoint slots must be zero, one, or two"
            )
        native_world_controls = _science_native_world_controls_v1(value)
        native_control = _science_native_control_v1(value)
        raw_logical_world_uids = tuple(
            cells[cell_id].get("science_logical_world_id")
            for cell_id in cell_ids
        )
        logical_world_uids = (
            tuple(int(value) for value in raw_logical_world_uids)
            if all(type(value) is int for value in raw_logical_world_uids)
            else ()
        )
        checkpoints = tuple(
            int(value) for value in execution.get("checkpoint_ticks", ())
        )
        restore_ticks = tuple(
            int(value) for value in execution.get("deliberate_restore_ticks", ())
        )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=cell_ids,
            checkpoint_milestones=(),
            parameter_arrays=tuple(
                (
                    name,
                    tuple(cells[cell_id]["parameters"][name] for cell_id in cell_ids),
                )
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (
                    name,
                    tuple(int(cells[cell_id]["seeds"][name]) for cell_id in cell_ids),
                )
                for name in seed_names
            ),
            checkpoint_storage_mode=str(
                execution.get("checkpoint_storage_mode", "disk")
            ),
            reporting_enabled=bool(execution.get("reporting_enabled", True)),
            execution_route_sha256=self.execution_route_contract(
                value
            ).execution_route_sha256,
            total_ticks=int(execution["total_ticks"]),
            start_next_tick=int(execution.get("start_next_tick", 0)),
            checkpoint_ticks=checkpoints,
            deliberate_restore_tick=(restore_ticks[0] if restore_ticks else None),
            deliberate_restore_ticks=restore_ticks,
            acs_checkpoint_slots=acs_checkpoint_slots,
            native_streamed_checkpoint=True,
            resident_owner_span_ticks=int(control["resident_owner_span_ticks"]),
            native_dispatch_window_ticks=int(control["native_dispatch_window_ticks"]),
            report_every_ticks=control.get("report_every_ticks"),
            science_sample_interval_ticks=control.get("science_sample_interval_ticks"),
            reducer_every_ticks=control.get("reducer_every_ticks"),
            archive_every_ticks=control.get("archive_every_ticks"),
            event_window_capacity=int(control["event_window_capacity"]),
            source_window_ticks=int(control["source_window_ticks"]),
            tick_native_control=dict(control),
            native_control=(
                None if native_control is None else dict(native_control)
            ),
            # Resident/runtime identities must remain unique logical-world
            # subjects.  Paired RNG identity is carried separately in each
            # generic world-control row and therefore never aliases evidence.
            native_rng_world_uids=(
                logical_world_uids if native_world_controls else ()
            ),
            native_world_controls=tuple(
                dict(control) for control in native_world_controls
            ),
        )

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        value = _payload(plan)
        reasons: list[UnsupportedReasonV1] = []
        try:
            contract = _native_contract_v1(request)
            artifact = _artifact_unit_v1(value, validator=self._artifact_validator)
            selection = artifact["selection"]["selection"]
            if request.profile_id != self.profile_id:
                raise CampaignProviderError("request profile differs")
            shared_policy = validate_shared_w32_plan_v1(
                value, temporal_k=contract.temporal_k
            )
            if shared_policy is None and len(value.get("physical_batches", ())) != 1:
                raise CampaignProviderError("route requires exactly one physical batch")
            if selection["selected_batch_size"] != contract.B:
                raise CampaignProviderError("sealed B differs from compiled physical B")
            if selection["temporal_k"] != contract.temporal_k:
                raise CampaignProviderError("sealed K differs from compiled temporal K")
            mode = _uniform_value_v1(request, "cuda_native.execution_mode")
            if selection["execution_mode"] != mode:
                raise CampaignProviderError("sealed execution mode differs")
            event_transport = _uniform_value_v1(request, "cuda_native.event_transport")
            if (
                event_transport not in EVENT_TRANSPORTS_V2
                or artifact["selection"]["event_transport"] != event_transport
            ):
                raise CampaignProviderError("sealed event transport differs")
            science_observation_mode = _uniform_value_v1(
                request, "cuda_native.science_observation_mode"
            )
            if science_observation_mode not in SCIENCE_OBSERVATION_MODES_V1:
                raise CampaignProviderError(
                    "science observation mode is not supported by the V2 provider"
            )
            science_materialization = value.get("science_materialization")
            native_world_controls = _science_native_world_controls_v1(value)
            if tuple(request.native_world_controls) != native_world_controls:
                raise CampaignProviderError(
                    "request native-world controls differ from the applied plan controls"
                )
            native_control = _science_native_control_v1(value)
            if native_control is None:
                if request.native_control is not None:
                    raise CampaignProviderError(
                        "request carries an unbound native-control receipt"
                    )
            elif request.native_control != native_control:
                raise CampaignProviderError(
                    "request native-control receipt differs from the applied plan receipt"
                )
            if (
                native_world_controls
                and event_transport != DIRECT_SPARSE_EVENT_TRANSPORT_V3
            ):
                raise CampaignProviderError(
                    "science native-world controls require direct-sparse V3 transport"
                )
            if science_observation_mode == SCIENCE_OBSERVATION_TYPED_SIDECAR_V1:
                if event_transport != DIRECT_SPARSE_EVENT_TRANSPORT_V3:
                    raise CampaignProviderError(
                        "typed science capture requires direct-sparse V3 transport"
                    )
                if not isinstance(science_materialization, Mapping):
                    raise CampaignProviderError(
                        "typed science capture lacks explicit materialization metadata"
                    )
                if (
                    science_materialization.get("observation_mode")
                    != SCIENCE_OBSERVATION_TYPED_SIDECAR_V1
                    or science_materialization.get("engineering_capture") is not True
                    or science_materialization.get("observation_abi")
                    != "eiel_science_observation_record_v1"
                ):
                    raise CampaignProviderError(
                        "typed science capture is not bound to the engineering observation ABI"
                    )
                if contract.temporal_k != 1:
                    raise CampaignProviderError(
                        "typed science capture currently requires temporal K=1"
                    )
            if science_observation_mode == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1:
                if event_transport != DIRECT_SPARSE_EVENT_TRANSPORT_V3:
                    raise CampaignProviderError(
                        "compact science capture requires direct-sparse V3 transport"
                    )
                if not native_world_controls:
                    raise CampaignProviderError(
                        "compact science capture requires generic native-world controls"
                    )
                if not isinstance(science_materialization, Mapping):
                    raise CampaignProviderError(
                        "compact science capture lacks explicit materialization metadata"
                    )
                if (
                    science_materialization.get("observation_mode")
                    != SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
                    or science_materialization.get("engineering_capture") is not True
                    or science_materialization.get("observation_abi")
                    != "eiel_cuda_v2_compact_science_capture_v1"
                ):
                    raise CampaignProviderError(
                        "compact science capture is not bound to its canonical observation ABI"
                    )
            if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3:
                runtime_manifest = artifact["manifest"].get("runtime")
                if (
                    not isinstance(runtime_manifest, Mapping)
                    or runtime_manifest.get("production_owner")
                    != "CudaFullEvolutionDirectSparseOwnerV3"
                    or runtime_manifest.get("horizon_dependent_resident_bytes") != 0
                ):
                    raise CampaignProviderError(
                        "native manifest lacks direct-sparse owner authority"
                    )
            if request.native_streamed_checkpoint is not True:
                raise CampaignProviderError(
                    "route requires native checkpointing"
                )
            expected_acs_slots = value.get("execution", {}).get(
                "acs_checkpoint_slots", 0
            )
            if (
                type(expected_acs_slots) is not int
                or expected_acs_slots not in {0, 1, 2}
                or request.acs_checkpoint_slots != expected_acs_slots
            ):
                raise CampaignProviderError(
                    "route ACS checkpoint slots differ from the compiled plan"
                )
            if request.checkpoint_storage_mode != "disk":
                raise CampaignProviderError("route requires disk checkpoint storage")
            if request.native_dispatch_window_ticks != contract.temporal_k:
                raise CampaignProviderError("native dispatch span must equal sealed K")
            if request.resident_owner_span_ticks != contract.temporal_k:
                raise CampaignProviderError("resident owner span must equal sealed K")
            if request.source_window_ticks != contract.temporal_k:
                raise CampaignProviderError("source span must equal sealed K")
            expected_event_capacity = (
                int(selection["initial_capacity_rows_per_world"])
                if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                else 1024 * contract.temporal_k
            )
            if request.event_window_capacity != expected_event_capacity:
                raise CampaignProviderError(
                    "event capacity differs from the sealed transport capacity"
                )
            target = contract.start_next_tick + contract.total_ticks
            if (
                request.checkpoint_ticks != tuple(sorted(set(request.checkpoint_ticks)))
                or not request.checkpoint_ticks
                or request.checkpoint_ticks[-1] != target
                or any(
                    tick <= contract.start_next_tick or tick > target
                    for tick in request.checkpoint_ticks
                )
            ):
                raise CampaignProviderError("absolute checkpoint schedule is invalid")
            self.execution_route_contract(value, request=request)
        except (CampaignProviderError, OSError, TypeError, ValueError) as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "cuda_full_evolution_v2_binding_invalid",
                    "execution",
                    str(exc),
                    self.profile_id,
                )
            )
        return tuple(reasons)

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        value = _payload(plan)
        artifact = _artifact_unit_v1(value, validator=self._artifact_validator)
        selected = artifact["selection"]
        event_transport = str(selected["event_transport"])
        selection_payload = selected["selection"]
        execution = value.get("execution")
        acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
            raise CampaignProviderError(
                "CUDA Native V2 route ACS checkpoint slots must be zero, one, or two"
            )
        try:
            shared_policy = validate_shared_w32_plan_v1(
                value,
                temporal_k=int(selection_payload["temporal_k"]),
            )
        except CudaFullEvolutionV2SharedRouteError as exc:
            raise CampaignProviderError(str(exc)) from exc
        native_control = _science_native_control_v1(value)
        science_observation_mode = (
            str(
                _uniform_value_v1(
                    request, "cuda_native.science_observation_mode"
                )
            )
            if request is not None
            else _plan_science_observation_mode_v1(value)
        )
        compact_science_capture = (
            science_observation_mode
            == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
        )
        if (
            native_control is not None
            and event_transport != DIRECT_SPARSE_EVENT_TRANSPORT_V3
        ):
            raise CampaignProviderError(
                "science native controls require direct-sparse V3 transport"
            )
        host_execution = host_execution_component_identity_v1()
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "absolute_tick_time_contract",
                    "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                    "cursor_semantics": "committed_half_open_[0,next_tick)",
                    "arbitrary_positive_terminal_tick": True,
                    "arbitrary_checkpoint_ticks": True,
                    "fresh_owner_partial_tick_restore": True,
                    "legacy_horizon_authority": False,
                },
                {
                    "component_id": "cuda_full_evolution_v2_native_transition",
                    "route_id": CUDA_NATIVE_ROUTE_V2,
                    "native_abi": CUDA_NATIVE_ABI_V1,
                    "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
                    "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                },
                host_execution,
                *(
                    (
                        {
                            "component_id": (
                                "cuda_full_evolution_v2_science_native_control"
                            ),
                            "schema_version": native_control["schema_version"],
                            "control_vector": dict(
                                native_control["control_vector"]
                            ),
                            "capability_mask": native_control["capability_mask"],
                            "control_digest": dict(
                                native_control["control_digest"]
                            ),
                            "applied": True,
                        },
                    )
                    if native_control is not None
                    else ()
                ),
                {
                    "component_id": "cuda_full_evolution_v2_artifact_unit",
                    "binary_sha256": artifact["binary_sha256"],
                    "manifest_file_sha256": artifact["manifest_file_sha256"],
                    "manifest_sha256": artifact["manifest_sha256"],
                },
                {
                    "component_id": "cuda_full_evolution_v2_device_selection",
                    "selection_receipt_sha256": selected["receipt_sha256"],
                    "selection_file_sha256": selected["file_sha256"],
                    "device_identity": selected["device_identity"],
                    "B": selected["selection"]["selected_batch_size"],
                    "temporal_k": selected["selection"]["temporal_k"],
                    "execution_mode": selected["selection"]["execution_mode"],
                    "event_transport": event_transport,
                    "initial_capacity_rows_per_world": selection_payload.get(
                        "initial_capacity_rows_per_world"
                    ),
                    "horizon_dependent_resident_bytes": (
                        selection_payload.get("horizon_dependent_resident_bytes")
                        if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                        else None
                    ),
                },
                {
                    "component_id": "cuda_full_evolution_v2_event_transport",
                    "event_transport": event_transport,
                    "execution_mode_is_orthogonal": True,
                    "wire_abi": (
                        "eiel_full_evolution_cuda_sparse_lifecycle_event_v3"
                        if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                        else "eiel_full_evolution_cuda_dense_event_v2"
                    ),
                    "horizon_dependent_resident_bytes": (
                        0
                        if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                        else None
                    ),
                },
                *(
                    (
                        {
                            "component_id": (
                                "cuda_full_evolution_v2_shared_gpu_resource_policy"
                            ),
                            **shared_policy,
                        },
                    )
                    if shared_policy is not None
                    else ()
                ),
                {
                    "component_id": "cuda_full_evolution_v2_checkpoint_occurrence",
                    "checkpoint_format_id": CHECKPOINT_SCHEMA_V2,
                    "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                    "attempt_authority_source": ATTEMPT_AUTHORITY_NAME,
                    "shadow_identity_is_authority": False,
                    "composition_adapter_required": True,
                },
                *(
                    (
                        {
                            "component_id": "cuda_full_evolution_v2_acs2_checkpoint_transport",
                            "acs_checkpoint_slots": acs_checkpoint_slots,
                            "capture_mode": "direct_resident_cuda_rows_v1",
                            "worker_boundary": "path_and_scalar_metadata_only",
                            "worker_accepts_cuda_objects": False,
                            "worker_accepts_callbacks": False,
                            "worker_full_state_bytes_materialized": 0,
                            "capture_pool": "one_shared_two_slab_pool_not_per_slot",
                            "pinned_slab_count": 2,
                            "producer_full_state_host_bytes_materialized": 0,
                            "producer_compacted_cuda_bytes_materialized": 0,
                            **(
                                {
                                    "compact_science_frontier_binding": (
                                        "acs_capture_descriptor_and_provider_checkpoint_record_v1"
                                    )
                                }
                                if compact_science_capture
                                else {}
                            ),
                            "publication": "strict_noreplace_post_fsync_readback_v1",
                        },
                    )
                    if acs_checkpoint_slots
                    else ()
                ),
                {
                    "component_id": (
                        "cuda_full_evolution_v2_science_continuation_prerequisite"
                    ),
                    "observation_mode": (
                        science_observation_mode
                    ),
                    "contract_sha256": artifact["science_continuation_contract"][
                        "contract_sha256"
                    ],
                    "required_checkpoint_adapter_capability": artifact[
                        "science_continuation_contract"
                    ]["required_checkpoint_adapter_capability"],
                    "required_canonical_identity_types": artifact[
                        "science_continuation_contract"
                    ]["required_canonical_identity_types"],
                    "checkpoint_adapter_present": False,
                    "checkpoint_adapter_validated": False,
                    "base_checkpoint_science_continuable": False,
                    "science_route_qualified": False,
                    "engineering_capture": (
                        science_observation_mode
                        in {
                            SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
                            SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1,
                        }
                    ),
                    "observation_abi": (
                        "eiel_science_observation_record_v1"
                        if science_observation_mode
                        == SCIENCE_OBSERVATION_TYPED_SIDECAR_V1
                        else (
                            "eiel_cuda_v2_compact_science_capture_v1"
                            if compact_science_capture
                            else None
                        )
                    ),
                },
                *(
                    (
                        {
                            "component_id": (
                                "cuda_full_evolution_v2_compact_science_async_publication"
                            ),
                            "workers": self._compact_science_async_workers,
                            "pending_slots": (
                                self._compact_science_async_pending_slots
                            ),
                            "worker_boundary": (
                                "immutable_tensor_free_host_capsule_v1"
                            ),
                            "worker_accepts_cuda_objects": False,
                            "worker_accepts_runtime_owner": False,
                            "fifo_publication": True,
                            "checkpoint_frontier_requires_durable_prefix": True,
                            **(
                                {
                                    "acs2_frontier_transaction_binding": (
                                        "acs_capture_descriptor_and_provider_checkpoint_record_v1"
                                    )
                                }
                                if acs_checkpoint_slots
                                else {}
                            ),
                            "partial_final_roots_visible": False,
                            "publication_commit": (
                                "linux_renameat2_strict_noreplace_v1"
                            ),
                            "worker_verifies_fsynced_payload_bytes": True,
                            "lifecycle_inventory_unit": "sealed_dispatch_tree_v1",
                            "device_to_host_overlap": False,
                            "direct_sparse_durable_publication_overlap": False,
                            "overlapped_scope": (
                                "compact_projection_compression_and_stage_fsync_v1"
                            ),
                        },
                    )
                    if compact_science_capture
                    else ()
                ),
                {
                    "component_id": "cuda_full_evolution_v2_durable_event_sink",
                    "persistence_schema": (
                        DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3
                        if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                        else EVENT_PERSISTENCE_SCHEMA_V1
                    ),
                    "acknowledge_only_after_receipt_fsync": True,
                    "owner_issued_prepared_capability_required": (
                        event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
                    ),
                    "compact_raw_envelope_retention": (
                        "exact_r0_until_terminal_lifecycle_release_v1"
                        if compact_science_capture
                        else None
                    ),
                },
            ),
            required_obligations=(
                "cold_start",
                "first_tick",
                "durable_frontier_publication",
                "exact_absolute_tick_checkpoint_export",
                "disk_checkpoint_reload",
                "fresh_owner_restore",
                "terminal_settlement",
                *(
                    (
                        "shared_gpu_peak_vram_admission",
                        "shared_gpu_lane_exclusion",
                        "uniform_multibatch_partition_binding",
                    )
                    if shared_policy is not None
                    else ()
                ),
                *(
                    (
                        "async_checkpoint_backpressure",
                        "async_checkpoint_failure_latch",
                        "acs2_host_only_capture",
                    )
                    if acs_checkpoint_slots
                    else ()
                ),
                *(
                    (
                        "async_compact_science_backpressure",
                        "async_compact_science_failure_latch",
                        "compact_science_host_only_capture",
                        "compact_science_checkpoint_prefix_drain",
                    )
                    if compact_science_capture
                    else ()
                ),
            ),
        )
        if (
            request is not None
            and request.execution_route_sha256 is not None
            and request.execution_route_sha256 != contract.execution_route_sha256
        ):
            raise CampaignProviderError("CUDA Native V2 route binding mismatch")
        return contract

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1],
    ) -> dict[str, Any]:
        value = _payload(plan)
        artifact = _artifact_unit_v1(value, validator=self._artifact_validator)
        try:
            shared_policy = validate_shared_w32_plan_v1(value)
        except CudaFullEvolutionV2SharedRouteError as exc:
            raise CampaignProviderError(str(exc)) from exc
        if shared_policy is None and len(requests) != 1:
            raise CampaignProviderError("CUDA Native V2 requires one bound request")
        if shared_policy is not None:
            expected_batches = value.get("physical_batches")
            if (
                not isinstance(expected_batches, list)
                or len(requests) != len(expected_batches)
                or [request.batch_id for request in requests]
                != [str(row.get("batch_id")) for row in expected_batches]
            ):
                raise CampaignProviderError(
                    "shared W32 resource requests differ from the ordered partition"
                )
        if not requests:
            raise CampaignProviderError("CUDA Native V2 requires bound requests")
        request = requests[0]
        contract = _native_contract_v1(request)
        if shared_policy is not None:
            contracts = tuple(_native_contract_v1(candidate) for candidate in requests)
            if any(
                candidate.B != contract.B
                or candidate.temporal_k != contract.temporal_k
                or candidate.total_ticks != contract.total_ticks
                or candidate.start_next_tick != contract.start_next_tick
                for candidate in contracts
            ) or any(
                candidate.acs_checkpoint_slots != request.acs_checkpoint_slots
                for candidate in requests
            ):
                raise CampaignProviderError(
                    "shared W32 physical requests are not one uniform execution shape"
                )
            try:
                validate_shared_w32_plan_v1(
                    value,
                    temporal_k=contract.temporal_k,
                )
            except CudaFullEvolutionV2SharedRouteError as exc:
                raise CampaignProviderError(str(exc)) from exc
        sealed_selection = artifact["selection"]["selection"]
        if sealed_selection["selected_batch_size"] != contract.B:
            raise CampaignProviderError("resource estimate differs from sealed B")
        event_transport = artifact["selection"]["event_transport"]
        direct_sparse = event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
        native_control = _science_native_control_v1(value)
        if (
            native_control is not None
            and not direct_sparse
        ):
            raise CampaignProviderError(
                "science native controls require direct-sparse V3 transport"
            )
        checkpoints = max(1, len(request.checkpoint_ticks))
        if direct_sparse:
            compact_science_capture = (
                _plan_science_observation_mode_v1(value)
                == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
            )
            estimate = _direct_sparse_resource_estimate_v3(
                contract,
                checkpoint_boundaries=checkpoints,
                sealed_selection=sealed_selection,
                device_selection=artifact["selection"],
                acs_checkpoint_slots=request.acs_checkpoint_slots,
                compact_science_capture=compact_science_capture,
                compact_snapshot_boundaries=len(
                    _science_snapshot_ticks_v1(value, request)
                ),
                progressive_storage_guard=(
                    shared_policy is None
                    and bool(request.native_world_controls)
                    and request.acs_checkpoint_slots == 0
                    and _plan_science_observation_mode_v1(value)
                    == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
                ),
                compact_async_workers=self._compact_science_async_workers,
                compact_async_pending_slots=(
                    self._compact_science_async_pending_slots
                ),
            )
            if compact_science_capture:
                try:
                    snapshot_implementation = (
                        self._native_snapshot_implementation_selector()
                    )
                except Exception as exc:
                    raise CampaignProviderError(
                        "compact science snapshot implementation selection "
                        f"failed during resource accounting: {exc}"
                    ) from exc
                if snapshot_implementation not in _native_snapshot_publication.NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1:
                    raise CampaignProviderError(
                        f"{COMPACT_SCIENCE_SNAPSHOT_IMPLEMENTATION_ENV_V1} "
                        "must select exactly python_v1, native_v1, or native_capsule_v1"
                    )
                if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
                    snapshot_implementation
                ):
                    estimate.update(
                        _native_snapshot_resource_accounting_v1(
                            contract,
                            slot_count=self._native_snapshot_slots,
                            implementation=snapshot_implementation,
                        )
                    )
            if shared_policy is not None:
                estimate = _aggregate_shared_w32_resource_estimate_v1(
                    estimate,
                    physical_batch_count=len(requests),
                    logical_world_count=sum(len(item.cell_ids) for item in requests),
                )
                try:
                    validate_shared_w32_plan_v1(
                        value,
                        temporal_k=contract.temporal_k,
                        vram_bytes_upper_bound=int(
                            estimate["vram_bytes_upper_bound"]
                        ),
                    )
                except CudaFullEvolutionV2SharedRouteError as exc:
                    raise CampaignProviderError(str(exc)) from exc
            if native_control is not None:
                estimate["native_control"] = dict(native_control)
            return estimate
        ledger = _cuda_v2_layout.estimate_cuda_native_resources_v2(
            contract, include_reference_scratch=False
        )
        dispatches = (
            contract.total_ticks + contract.temporal_k - 1
        ) // contract.temporal_k
        event_bytes = (
            contract.B * 1024 * 48 * contract.total_ticks + contract.B * 4 * dispatches
        )
        checkpoint_bytes = checkpoints * (
            contract.B
            * _cuda_v2_layout.state_layout_v2().canonical_checkpoint_bytes_per_world
            + 128 * 1024
        )
        checkpoint_payload_bytes = (
            contract.B
            * _cuda_v2_layout.state_layout_v2().canonical_checkpoint_bytes_per_world
        )
        estimate = {
            "resource_estimator_id": "cuda_full_evolution_v2_sealed_bk_v1",
            "vram_bytes_upper_bound": ledger.total_bytes,
            "wall_seconds_upper_bound": cuda_full_evolution_v2_wall_seconds_upper_bound_v1(
                contract.total_ticks
            ),
            "disk_bytes_upper_bound": event_bytes + checkpoint_bytes + 64 * 1024**2,
            "physical_batch_cells": contract.B,
            "sealed_total_vram_bytes": artifact["selection"]["capacity_basis"][
                "total_vram_bytes"
            ],
            "sealed_fixed_reserve_bytes": artifact["selection"]["capacity_basis"][
                "fixed_reserve_bytes"
            ],
            "event_transport": event_transport,
            "horizon_dependent_resident_bytes": None,
            "acs_checkpoint_slots": request.acs_checkpoint_slots,
            "checkpoint_capture_mode": (
                "direct_resident_cuda_rows_v1"
                if request.acs_checkpoint_slots
                else "synchronous_v1"
            ),
            "acs_capture_pool_count": 1 if request.acs_checkpoint_slots else 0,
            "acs_pinned_slab_count": 2 if request.acs_checkpoint_slots else 0,
            "acs_pinned_host_bytes_upper_bound": (
                2 * ACS2_CAPTURE_CHUNK_BYTES_V1
                if request.acs_checkpoint_slots
                else 0
            ),
            "acs_pending_capture_files_upper_bound": request.acs_checkpoint_slots,
            "acs_pending_capture_file_bytes_upper_bound": (
                request.acs_checkpoint_slots * checkpoint_payload_bytes
            ),
            "acs_producer_full_state_host_bytes": 0,
            "acs_producer_compacted_cuda_bytes": 0,
            "acs_worker_full_state_bytes": 0,
            "acs_worker_cuda_references": 0,
        }
        if shared_policy is not None:
            estimate = _aggregate_shared_w32_resource_estimate_v1(
                estimate,
                physical_batch_count=len(requests),
                logical_world_count=sum(len(item.cell_ids) for item in requests),
            )
            try:
                validate_shared_w32_plan_v1(
                    value,
                    temporal_k=contract.temporal_k,
                    vram_bytes_upper_bound=int(estimate["vram_bytes_upper_bound"]),
                )
            except CudaFullEvolutionV2SharedRouteError as exc:
                raise CampaignProviderError(str(exc)) from exc
        if native_control is not None:
            estimate["native_control"] = dict(native_control)
        return estimate

    def crca_source_identity_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        launchpad_source_identity: Mapping[str, Any],
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project exact current locks for CRCA without creating authority."""

        artifact = _artifact_unit_v1(plan, validator=self._artifact_validator)
        git = launchpad_source_identity.get("git")
        if not isinstance(git, Mapping):
            raise CampaignProviderError("CRCA projection lacks exact Git identity")
        commit = git.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise CampaignProviderError("CRCA projection lacks a full Git commit")
        dirty = (
            None
            if git.get("working_tree_clean") is True
            else _sha256_text(
                git.get("working_tree_status_sha256"),
                name="working tree status digest",
            )
        )
        manifest = artifact["manifest"]
        runtime_environment = preflight.get("runtime_environment")
        return {
            "source_commit": commit,
            "source_tree_identity": {
                "campaign_spec_sha256": launchpad_source_identity.get(
                    "campaign_spec_sha256"
                ),
                "compiled_plan_sha256": launchpad_source_identity.get(
                    "compiled_plan_sha256"
                ),
                "bound_plan_sha256": launchpad_source_identity.get("bound_plan_sha256"),
                "working_tree_status_sha256": git.get("working_tree_status_sha256"),
            },
            "dirty_patch_sha256": dirty,
            "toolchain": {
                "native_manifest_python": manifest.get("python"),
                "native_manifest_build": manifest.get("build"),
                "native_manifest_target_sm": manifest.get("target_sm"),
                "native_manifest_sha256": artifact["manifest_sha256"],
                "science_native_control": _science_native_control_v1(_payload(plan)),
            },
            "environment": {
                "preflight_runtime_environment": (
                    dict(runtime_environment)
                    if isinstance(runtime_environment, Mapping)
                    else {}
                ),
                "native_runtime_contract": manifest.get("runtime"),
            },
            "device_conformance_envelope": artifact["selection"]["device_identity"],
        }

    def crca_assurance_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        route = self.execution_route_contract(plan)
        plan_payload = _payload(plan)
        lifecycle = plan_payload.get("artifact_lifecycle")
        qualification_candidate = (
            isinstance(lifecycle, Mapping)
            and lifecycle.get("lifecycle_profile") == QUALIFICATION
        )
        return {
            "intended_claim_id": (
                "TARGETED_DELTA_QUALIFICATION_CANDIDATE_CUDA_NATIVE_V2"
                if qualification_candidate
                else "TECHNICAL_ONLY_PREQUAL_CUDA_NATIVE_V2"
            ),
            "qualified_envelope": None,
            "obligations": {
                "preflight": {
                    "ready_for_execution": preflight.get("ready_for_execution"),
                    "receipt_sha256": canonical_sha256(preflight),
                },
                "execution_route": route.payload(),
            },
            "authority_references": {
                "route_authority": {
                    "provider_id": self.provider_id,
                    "profile_id": self.profile_id,
                    "execution_route_sha256": route.execution_route_sha256,
                    "qualification_status": (
                        "candidate_unqualified"
                        if qualification_candidate
                        else "experimental"
                    ),
                }
            },
            "review_references": {},
            "initial_holds": {
                "formal_qualification": {
                    "status": (
                        "HOLD_QUALIFICATION_DECISION"
                        if qualification_candidate
                        else "HOLD_PREQUAL_ONLY"
                    ),
                    "reason": (
                        "targeted delta Qualification has not issued a decision"
                        if qualification_candidate
                        else "no targeted delta Qualification envelope"
                    ),
                }
            },
        }

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        return self._execute_with_bound_acs_runtime_v1(
            plan=plan,
            request=request,
            context=context,
            resume_checkpoint=None,
        )

    def resume_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return self._execute_with_bound_acs_runtime_v1(
            plan=plan,
            request=request,
            context=context,
            resume_checkpoint=checkpoint,
        )

    def _execute_with_bound_acs_runtime_v1(
        self,
        *,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
        resume_checkpoint: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute with the exact plan-bound AOT ACS runtime when required."""

        value = _payload(plan)
        binding = None
        if request.acs_checkpoint_slots and self._acs_require_native_pool:
            binding = _acs_native_runtime_artifact_binding_v1(
                value,
                required=True,
            )
        prior_extension = os.environ.get(
            ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1
        )
        if binding is not None:
            os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = str(
                binding["extension_path"]
            )
        try:
            return _execute_native_batch_v1(
                provider=self,
                plan=value,
                request=request,
                context=context,
                resume_checkpoint=resume_checkpoint,
            )
        finally:
            if binding is not None:
                if prior_extension is None:
                    os.environ.pop(
                        ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1,
                        None,
                    )
                else:
                    os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = (
                        prior_extension
                    )


def cuda_full_evolution_v2_materialized_resource_estimate_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Re-estimate one executable, already-materialized native plan.

    The authoring compiler cannot know the later explicit-world observation
    schedule. Science carrier lowering uses this provider-owned boundary to
    bind the same artifact/device/B/K accounting used by preflight and
    execution, including compact acquisition buffers and output.
    """

    value = _payload(plan)
    if value.get("resolved_profile") != PROFILE_ID:
        raise CampaignProviderError(
            "materialized resource estimation requires the CUDA Native V2 profile"
        )
    batches = value.get("physical_batches")
    if not isinstance(batches, list) or not batches:
        raise CampaignProviderError(
            "materialized resource estimation requires physical batches"
        )
    provisional = dict(value)
    provisional.pop("plan_sha256", None)
    provisional["plan_sha256"] = canonical_sha256(provisional)
    provider = CudaFullEvolutionV2ExecutionProviderV1()
    requests = tuple(
        provider.translate_batch(provisional, str(batch.get("batch_id")))
        for batch in batches
        if isinstance(batch, Mapping) and isinstance(batch.get("batch_id"), str)
    )
    if len(requests) != len(batches):
        raise CampaignProviderError(
            "materialized resource estimation found a malformed physical batch"
        )
    return dict(provider.estimate_resources(provisional, requests))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_durable(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise CampaignProviderError(
                    f"durable publication target already exists: {path}"
                ) from exc
            temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _seal_read_only_file_v1(path: Path) -> None:
    path.chmod(0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _jsonable_v1(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable_v1(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable_v1(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_v1(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


SCIENCE_OBSERVATION_SIDECAR_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_observation_sidecar_v1"
)
SCIENCE_OBSERVATION_ABI_V1: Final[str] = "eiel_science_observation_record_v1"


def _science_digest_v1(value: object, *, name: str) -> ContentDigest:
    if isinstance(value, ContentDigest):
        return value
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise CampaignProviderError(f"{name} must be an exact SHA-256 digest")
    try:
        return ContentDigest(str(value["algorithm"]), str(value["hex"]))
    except Exception as exc:
        raise CampaignProviderError(f"{name} is not a valid SHA-256 digest") from exc


def _science_capture_cells_v1(
    plan: Mapping[str, Any], request: ResidentBatchRequestV1
) -> dict[str, Mapping[str, Any]]:
    _batch, cell_ids, cells = _batch_and_cells_v1(plan, request.batch_id)
    selected = {cell_id: cells[cell_id] for cell_id in cell_ids}
    for cell_id, cell in selected.items():
        for field in (
            "science_semantic_inputs",
            "science_acquisition",
            "science_reducers",
            "science_semantic_world_key",
            "science_logical_world_id",
            "science_acquisition_sha256",
            "science_instrument_manifest_sha256",
            "science_crca_semantic_run_digest",
            "science_requested_observations",
        ):
            if field not in cell:
                raise CampaignProviderError(
                    f"typed science cell {cell_id} lacks {field}"
                )
        requested = cell.get("science_requested_observations")
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
            raise CampaignProviderError(
                f"typed science cell {cell_id} has an invalid observation contract"
            )
    return selected


def _native_world_control_set_hex_v1(
    plan: Mapping[str, Any], request: ResidentBatchRequestV1
) -> str:
    materialization = plan.get("science_materialization")
    value = (
        materialization.get("native_world_control_set_sha256")
        if isinstance(materialization, Mapping)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or value.get("algorithm") != "sha256"
        or not isinstance(value.get("hex"), str)
    ):
        raise CampaignProviderError(
            "science materialization lacks its native-world control-set digest"
        )
    digest = str(value["hex"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CampaignProviderError(
            "science materialization native-world control-set digest is invalid"
        )
    if len(request.native_world_controls) != len(request.cell_ids):
        raise CampaignProviderError(
            "native-world control count differs from the compact science batch"
        )
    return digest


def _native_snapshot_execution_publication_context_sha256_v1(
    *,
    plan: Mapping[str, Any],
    request: ResidentBatchRequestV1,
    context: Any,
    native_artifact: Mapping[str, Any],
    encoder_extension: Any,
    snapshot_implementation: str = "native_v1",
) -> str:
    """Bind immutable execution, output, evidence, and encoder identities."""

    manifest = getattr(encoder_extension, "manifest", None)
    output = manifest.get("output") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(output.get("sha256"), str)
        or not isinstance(manifest.get("manifest_sha256"), str)
    ):
        raise CampaignProviderError(
            "native snapshot encoder extension identity is incomplete"
        )
    lifecycle = getattr(context, "lifecycle_contract", None)
    repository_root = Path(__file__).resolve().parents[2]
    native_host_components = [
        {
            "component_id": component_id,
            "path": relative_path,
            "sha256": _sha256_file(repository_root / relative_path),
        }
        for component_id, relative_path in (
            NATIVE_SNAPSHOT_HOST_COMPONENT_PATHS_V1
        )
    ]
    body = {
        "schema_version": (
            "eiel_native_snapshot_execution_publication_context_v1"
        ),
        "provider_id": PROVIDER_ID,
        "snapshot_implementation": snapshot_implementation,
        "campaign_plan_sha256": plan.get("plan_sha256"),
        "execution_route_sha256": getattr(
            context, "execution_route_sha256", None
        ),
        "resident_request_sha256": canonical_sha256(request.payload()),
        "batch_id": request.batch_id,
        "ordered_cell_ids": list(request.cell_ids),
        "run_root": str(Path(context.run_root).resolve()),
        "batch_root": str(Path(context.batch_root).resolve()),
        "snapshot_root": str(
            (
                Path(context.batch_root)
                / "science-compact"
                / "snapshots"
            ).resolve()
        ),
        "artifact_family": (
            "cuda_full_evolution_v2.compact_science_snapshot"
        ),
        "native_evolution_artifact_sha256": native_artifact.get(
            "binary_sha256"
        ),
        "native_evolution_manifest_sha256": native_artifact.get(
            "manifest_sha256"
        ),
        "native_snapshot_encoder_artifact_path": str(
            Path(getattr(encoder_extension, "artifact_path")).resolve()
        ),
        "native_snapshot_encoder_artifact_sha256": output["sha256"],
        "native_snapshot_encoder_manifest_sha256": manifest[
            "manifest_sha256"
        ],
        "native_snapshot_host_components": native_host_components,
        "native_snapshot_host_components_sha256": canonical_sha256(
            native_host_components
        ),
        "artifact_lifecycle_contract_sha256": (
            lifecycle.get("contract_sha256")
            if isinstance(lifecycle, Mapping)
            else None
        ),
    }
    if any(
        body[name] is None
        for name in (
            "campaign_plan_sha256",
            "execution_route_sha256",
            "native_evolution_artifact_sha256",
            "native_evolution_manifest_sha256",
        )
    ):
        raise CampaignProviderError(
            "native snapshot execution/publication identity is incomplete"
        )
    return canonical_sha256(body)


def _record_compact_science_root_v1(
    context: Any,
    root: str | Path,
    *,
    artifact_family: str,
) -> Mapping[str, Any]:
    resolved = Path(root).resolve()
    files = tuple(sorted(path for path in resolved.iterdir() if path.is_file()))
    if not files or any(path.is_symlink() for path in files):
        raise CampaignProviderError(
            "compact science artifact root is empty or contains a symbolic link"
        )
    for path in files:
        _seal_read_only_file_v1(path)
    return context.record_artifact(
        resolved,
        artifact_family=artifact_family,
        retention_class="R3",
        release_predicate=(
            "canonical compact science observation retained for declared reducers"
        ),
        replacement_verified=False,
    )


def _settle_native_snapshot_publication_transaction_v1(
    *,
    pipeline: Any,
    token: Any,
    context: Any,
    producer_backpressure: bool,
    record_root: Callable[..., Mapping[str, Any]] = (
        _record_compact_science_root_v1
    ),
    reconcile_inventory: Callable[..., str] | None = None,
    remove_root: Callable[[str | Path], None] = (
        _native_snapshot_publication.remove_native_snapshot_root_v1
    ),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Publish and lifecycle-commit one exact native snapshot FIFO head."""

    published_root: Path | None = None
    inventory_state = "NOT_ATTEMPTED"
    artifact_family = "cuda_full_evolution_v2.compact_science_snapshot"
    phase_timing: dict[str, Any] = {}
    reconcile = (
        _reconcile_compact_science_inventory_commit_v1
        if reconcile_inventory is None
        else reconcile_inventory
    )
    try:
        started_ns = perf_counter_ns()
        staged_receipt = pipeline.result(
            token,
            producer_backpressure=producer_backpressure,
        )
        phase_timing["native_fifo_result_wait_ns"] = (
            perf_counter_ns() - started_ns
        )
        phase_timing["capture_metrics"] = dict(
            staged_receipt.get("capture_metrics", {})
        )
        phase_timing["encoder_timing"] = dict(
            staged_receipt.get("encoder_timing", {})
        )
        phase_timing["native_counters"] = dict(
            staged_receipt.get("native_counters", {})
        )
        phase_timing["worker_publication_timing"] = dict(
            staged_receipt.get("publication_timing", {})
        )
        started_ns = perf_counter_ns()
        compact_receipt = pipeline.publish(token, staged_receipt)
        phase_timing["atomic_publication_ns"] = perf_counter_ns() - started_ns
        published_root = Path(str(compact_receipt["path"]))
        compact_pointer = _compact_science_artifact_pointer_v1(compact_receipt)
        started_ns = perf_counter_ns()
        try:
            record_root(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
        except BaseException as record_error:
            inventory_state = reconcile(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
            if inventory_state != "COMMITTED_EXACT_DURABLE":
                raise CampaignProviderError(
                    "native snapshot lifecycle record failed with inventory "
                    f"state {inventory_state}"
                ) from record_error
        else:
            inventory_state = reconcile(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
            if inventory_state != "COMMITTED_EXACT_DURABLE":
                raise CampaignProviderError(
                    "native snapshot lifecycle record returned without a durable "
                    f"exact inventory commit: {inventory_state}"
                )
        phase_timing["manifest_evidence_lifecycle_registration_ns"] = (
            perf_counter_ns() - started_ns
        )
        started_ns = perf_counter_ns()
        pipeline.acknowledge_published(
            token,
            published_receipt=compact_receipt,
        )
        phase_timing["publication_acknowledgement_ns"] = (
            perf_counter_ns() - started_ns
        )
        return dict(compact_receipt), compact_pointer, phase_timing
    except BaseException as exc:
        if isinstance(
            exc, _native_snapshot_publication.NativeSnapshotVisibleRootError
        ):
            published_root = Path(exc.visible_root)
        if published_root is not None and inventory_state == "NOT_ATTEMPTED":
            inventory_state = reconcile(
                context,
                published_root,
                artifact_family=artifact_family,
            )
        preserve_root = inventory_state in {
            "COMMITTED_EXACT_DURABLE",
            "COMMITTED_VISIBLE_NOT_DURABLE_PRESERVE_ROOT",
            "COMMIT_UNKNOWN_PRESERVE_ROOT",
        }
        cleanup_error: BaseException | None = None
        if (
            not preserve_root
            and published_root is not None
            and os.path.lexists(published_root)
        ):
            try:
                remove_root(published_root)
            except BaseException as rollback_exc:
                cleanup_error = rollback_exc
        if (
            not preserve_root
            and published_root is not None
            and os.path.lexists(published_root)
        ):
            raise CampaignProviderError(
                "native snapshot settlement failed and canonical-root cleanup "
                "did not converge: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            ) from exc
        if isinstance(exc, CampaignProviderError):
            raise
        if isinstance(exc, Exception):
            raise CampaignProviderError(
                "native snapshot publication settlement failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise


def _expected_compact_science_inventory_row_v1(
    context: Any,
    root: str | Path,
    *,
    artifact_family: str,
) -> dict[str, Any]:
    return build_sealed_artifact_inventory_row_v1(
        run_root=Path(context.run_root),
        path=Path(root),
        artifact_family=artifact_family,
        retention_class="R3",
        owner_writer=(
            getattr(context, "provider_id", None) or "campaign-provider"
        ),
        release_predicate=(
            "canonical compact science observation retained for declared reducers"
        ),
        replacement_verified=False,
    )


def _reconcile_compact_science_inventory_commit_v1(
    context: Any,
    root: str | Path,
    *,
    artifact_family: str,
) -> str:
    """Classify and durably resolve a lifecycle record commit-unknown error."""

    try:
        expected = _expected_compact_science_inventory_row_v1(
            context,
            root,
            artifact_family=artifact_family,
        )
        manifest_path = Path(context.manifest_path)
        manifest = _json_object(
            manifest_path,
            name="batch manifest after compact lifecycle record failure",
        )
    except Exception:
        return "COMMIT_UNKNOWN_PRESERVE_ROOT"
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, list):
        return "NOT_COMMITTED_REMOVE_ROOT"
    same_path = [
        dict(row)
        for row in inventory
        if isinstance(row, Mapping) and row.get("path") == expected["path"]
    ]
    if not same_path:
        return "NOT_COMMITTED_REMOVE_ROOT"
    if (
        len(same_path) != 1
        or same_path[0] != expected
        or manifest.get("artifact_inventory_complete") is False
    ):
        return "COMMIT_UNKNOWN_PRESERVE_ROOT"
    try:
        directory_fd = os.open(
            manifest_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        return "COMMITTED_VISIBLE_NOT_DURABLE_PRESERVE_ROOT"
    return "COMMITTED_EXACT_DURABLE"


def _compact_science_index_v1(
    *,
    context: Any,
    request: ResidentBatchRequestV1,
    native_world_control_set_sha256: str,
    dispatch_receipts: Sequence[Mapping[str, Any]],
    snapshot_receipts: Sequence[Mapping[str, Any]],
    checkpoint_frontier: Mapping[str, Any] | None,
    terminal_tick: int,
) -> dict[str, Any]:
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_capture_index_v1",
        "observation_abi": "eiel_cuda_v2_compact_science_capture_v1",
        "batch_id": request.batch_id,
        "ordered_cell_ids": list(request.cell_ids),
        "native_world_control_set_sha256": native_world_control_set_sha256,
        "terminal_next_absolute_tick": terminal_tick,
        "dispatch_artifacts": [dict(value) for value in dispatch_receipts],
        "snapshot_artifacts": [dict(value) for value in snapshot_receipts],
        "checkpoint_frontier": (
            None if checkpoint_frontier is None else dict(checkpoint_frontier)
        ),
        "dropped_record_count": 0,
    }
    body["index_sha256"] = canonical_sha256(body)
    path = (
        Path(context.batch_root)
        / "science-compact"
        / "capture-index-terminal.json"
    )
    payload = canonical_bytes(body)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            raise CampaignProviderError(
                "compact science terminal index already exists with different content"
            )
    else:
        _write_bytes_durable(path, payload)
        _seal_read_only_file_v1(path)
    context.record_artifact(
        path,
        artifact_family="cuda_full_evolution_v2.compact_science_capture_index",
        retention_class="R5",
        release_predicate="canonical compact science artifact ownership index",
        replacement_verified=False,
    )
    return {**body, "path": str(path)}


def _compact_science_continuation_frontier_v1(
    *,
    counter_frontier: Mapping[str, Any],
    dispatch_receipts: Sequence[Mapping[str, Any]],
    snapshot_receipts: Sequence[Mapping[str, Any]],
    native_world_control_set_sha256: str,
) -> dict[str, Any]:
    value = {
        "schema_version": "eiel_cuda_v2_compact_science_continuation_frontier_v1",
        "next_absolute_tick": counter_frontier.get("next_absolute_tick"),
        "native_world_control_set_sha256": native_world_control_set_sha256,
        "counter_frontier": dict(counter_frontier),
        "dispatch_artifacts": [dict(item) for item in dispatch_receipts],
        "snapshot_artifacts": [dict(item) for item in snapshot_receipts],
    }
    value["frontier_sha256"] = canonical_sha256(value)
    return value


def _compact_science_artifact_pointer_v1(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    pointer = {
        key: receipt[key]
        for key in (
            "schema_version",
            "artifact_sha256",
            "path",
            "start_next_absolute_tick",
            "end_next_absolute_tick",
            "absolute_tick",
            "event_count",
            "row_count",
            "captured_world_count",
            "dropped_record_count",
            "source_sink_receipt_sha256",
        )
        if key in receipt
    }
    if not all(key in pointer for key in ("schema_version", "artifact_sha256", "path")):
        raise CampaignProviderError("compact science artifact pointer is incomplete")
    return pointer


def _compact_snapshot_timing_enabled_v1() -> bool:
    value = os.environ.get(COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1)
    if value in {None, "", "0"}:
        return False
    if value == "1":
        return True
    raise CampaignProviderError(
        f"{COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1} must be exactly 0 or 1"
    )


def _nearest_rank_ns_v1(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise CampaignProviderError("snapshot timing distribution is empty")
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _snapshot_duration_distribution_v1(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise CampaignProviderError("snapshot timing distribution is empty")
    ordered = tuple(sorted(int(value) for value in values))
    return {
        "sample_count": len(ordered),
        "minimum_ns": ordered[0],
        "p50_ns": _nearest_rank_ns_v1(ordered, 0.50),
        "p95_ns": _nearest_rank_ns_v1(ordered, 0.95),
        "maximum_ns": ordered[-1],
        "sum_ns": sum(ordered),
        "mean_ns": sum(ordered) / len(ordered),
    }


def _compact_snapshot_timing_summary_v1(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    implementations = {
        str(sample.get("implementation", "python_v1")) for sample in samples
    }
    if len(implementations) > 1 or any(
        value not in _native_snapshot_publication.NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1
        for value in implementations
    ):
        raise CampaignProviderError(
            "snapshot timing samples contain mixed or invalid implementations"
        )
    implementation = next(iter(implementations), "python_v1")
    phase_names = sorted(
        {
            str(name)
            for sample in samples
            for name in (
                sample.get("phases", {}).keys()
                if isinstance(sample.get("phases"), Mapping)
                else ()
            )
        }
    )
    phase_distributions: dict[str, Any] = {}
    phase_cpu_time_distributions: dict[str, Any] = {}
    phase_cuda_event_time_distributions: dict[str, Any] = {}
    for name in phase_names:
        values = [
            int(sample["phases"][name]["wall_time_ns"])
            for sample in samples
            if isinstance(sample.get("phases"), Mapping)
            and isinstance(sample["phases"].get(name), Mapping)
        ]
        phase_distributions[name] = _snapshot_duration_distribution_v1(values)
        cpu_values = [
            int(sample["phases"][name]["cpu_time_ns"])
            for sample in samples
            if isinstance(sample.get("phases"), Mapping)
            and isinstance(sample["phases"].get(name), Mapping)
            and type(sample["phases"][name].get("cpu_time_ns")) is int
        ]
        if cpu_values:
            phase_cpu_time_distributions[name] = (
                _snapshot_duration_distribution_v1(cpu_values)
            )
        cuda_event_values = [
            int(sample["phases"][name]["cuda_event_time_ns"])
            for sample in samples
            if isinstance(sample.get("phases"), Mapping)
            and isinstance(sample["phases"].get(name), Mapping)
            and type(sample["phases"][name].get("cuda_event_time_ns")) is int
        ]
        if cuda_event_values:
            phase_cuda_event_time_distributions[name] = (
                _snapshot_duration_distribution_v1(cuda_event_values)
            )
    metric_names = (
        "producer_critical_path_time_ns",
        "full_durable_completion_time_ns",
        "publication_lag_time_ns",
        "gpu_sync_host_wait_upper_bound_ns",
        "evolution_producer_blocking_time_ns",
    )
    duration_distributions = {
        name: _snapshot_duration_distribution_v1(
            [int(sample[name]) for sample in samples]
        )
        for name in metric_names
        if samples
    }
    counters: dict[str, int | float] = {}
    for name in (
        "row_count",
        "captured_world_count",
        "active_agent_count",
        "d2h_bytes",
        "uncompressed_bytes",
        "compressed_bytes",
    ):
        values = [
            sample.get("counters", {}).get(name)
            for sample in samples
            if isinstance(sample.get("counters"), Mapping)
        ]
        if values and all(type(value) in {int, float} for value in values):
            counters[f"{name}_sum"] = sum(values)
            counters[f"{name}_maximum"] = max(values)
    if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
        implementation
    ):
        timing_boundary = {
            "producer_critical_path": (
                "snapshot boundary entry through pinned D2H capture admission "
                "and bounded-service handoff, plus the exact CUDA-event duration "
                "that serializes the evolution stream and any explicit FIFO "
                "backpressure settlement; the host-only handoff remains in the "
                "native_capture_and_submit phase"
            ),
            "full_durable_completion": (
                "snapshot boundary entry through native encode/compression, "
                "stage fsync, atomic no-replace final publication, parent "
                "directory fsync, lifecycle registration, pointer append, and "
                "FIFO acknowledgement"
            ),
            "publication_lag": (
                "full durable completion minus producer handoff; bounded by the "
                "fixed resident slot pool and explicit producer backpressure"
            ),
            "gpu_sync_host_wait": (
                "producer host handoff plus the reported CUDA-event copy duration; "
                "the event can include earlier work on the bound evolution stream"
            ),
            "evolution_producer_blocking": (
                "producer capture/handoff, serialized D2H CUDA-event duration, "
                "and provider-thread FIFO settlement; background native encoding "
                "lag is excluded until its explicit drain or backpressure boundary"
            ),
        }
        memory_values = [
            sample.get("memory")
            for sample in samples
            if isinstance(sample.get("memory"), Mapping)
        ]

        def maximum_memory_value(name: str) -> int | None:
            values = [
                int(value[name])
                for value in memory_values
                if type(value.get(name)) is int
            ]
            return max(values) if values else None

        memory_measurement = {
            "process_rss": (
                "not sampled on the asynchronous native worker path; bounded "
                "slot bytes and service high-water counters are authoritative"
            ),
            "gpu": (
                "resident simulation allocation is unchanged; capture event "
                "telemetry accounts for the selected D2H operation"
            ),
            "pinned_host_bytes": maximum_memory_value(
                "resident_pinned_bytes"
            ),
            "native_heap_bytes": None,
            "snapshot_resident_buffer_count": max(
                (
                    int(value.get("core_service", {}).get("slot_count", 0))
                    for value in memory_values
                    if isinstance(value.get("core_service"), Mapping)
                ),
                default=0,
            ),
            "in_use_buffer_bytes_high_water": max(
                (
                    int(
                        value.get("core_service", {}).get(
                            "in_use_buffer_bytes_high_water", 0
                        )
                    )
                    for value in memory_values
                    if isinstance(value.get("core_service"), Mapping)
                ),
                default=0,
            ),
            "limitation": (
                "G5 uses dense state and metric sections in a synchronous "
                "producer capture into fixed reusable pinned slots; active-row "
                "compaction and first-class capsule publication remain later "
                "native data-plane gates"
            ),
        }
    else:
        timing_boundary = {
            "producer_critical_path": (
                "snapshot boundary entry through lifecycle registration and "
                "receipt-pointer append"
            ),
            "full_durable_completion": (
                "same as producer critical path for the synchronous V1 writer"
            ),
            "publication_lag": (
                "durable completion minus producer handoff; zero for synchronous V1"
            ),
            "gpu_sync_host_wait": (
                "host time spent at the two synchronous D2H boundaries; it is "
                "an upper bound because each wait can include earlier stream work"
            ),
            "evolution_producer_blocking": (
                "the complete synchronous snapshot boundary during which the "
                "producer cannot submit further evolution work"
            ),
        }
        memory_measurement = {
            "process_rss": (
                "cumulative ru_maxrss sampled immediately before and after each "
                "boundary; an unchanged value means no new process high-water"
            ),
            "gpu": (
                "current allocator bytes sampled before/after plus cumulative "
                "Torch CUDA allocator peaks when available"
            ),
            "pinned_host_bytes": 0,
            "native_heap_bytes": None,
            "limitation": (
                "V1 has no bounded native pool; exact per-snapshot allocation "
                "high-water arrives with the native service"
            ),
        }
    return {
        "schema_version": COMPACT_SCIENCE_SNAPSHOT_TIMING_SUMMARY_SCHEMA_V1,
        "claim_class": "TECHNICAL_ONLY",
        "enabled_by": COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1,
        "implementation": implementation,
        "sample_count": len(samples),
        "quantile_method": "nearest_rank_v1",
        "timing_boundary": timing_boundary,
        "duration_distributions": duration_distributions,
        "phase_distributions": phase_distributions,
        "phase_cpu_time_distributions": phase_cpu_time_distributions,
        "phase_cuda_event_time_distributions": (
            phase_cuda_event_time_distributions
        ),
        "aggregate_counters": counters,
        "memory_measurement": memory_measurement,
        "per_snapshot": [dict(sample) for sample in samples],
    }


def _compact_snapshot_timing_result_fields_v1(
    *,
    enabled: bool,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the default-off provider envelope exactly free of timing fields."""

    if not enabled:
        return {}
    return {
        "compact_science_snapshot_timing": (
            _compact_snapshot_timing_summary_v1(samples)
        )
    }


def _science_context_v1(cell: Mapping[str, Any]) -> Any:
    seeds = cell.get("seeds")
    if not isinstance(seeds, Mapping) or type(seeds.get("task")) is not int:
        raise CampaignProviderError("typed science cell lacks its native RNG world UID")
    logical_world_id = cell.get("science_logical_world_id")
    if type(logical_world_id) is not int or logical_world_id < 0:
        raise CampaignProviderError(
            "typed science cell lacks its explicit semantic logical world ID"
        )
    key = cell.get("science_semantic_world_key")
    if not isinstance(key, Mapping):
        raise CampaignProviderError("typed science cell semantic world key is invalid")
    acquisition = _science_digest_v1(
        cell.get("science_acquisition_sha256"),
        name="science acquisition identity",
    )
    manifest = _science_digest_v1(
        cell.get("science_instrument_manifest_sha256"),
        name="science instrument manifest identity",
    )
    crca = _science_digest_v1(
        cell.get("science_crca_semantic_run_digest"),
        name="science CRCA semantic run identity",
    )
    return _science_observation_projection.NativeCudaV2ObservationContextV1(
        instrument_manifest_sha256=manifest,
        crca_semantic_run_digest=crca,
        science_semantic_acquisition_sha256=acquisition,
        cell_id=str(cell["cell_id"]),
        logical_world_id=logical_world_id,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_world_key_v1"), dict(key)
        ),
    )


def _science_frontier_v1(
    records_by_cell: Mapping[str, Sequence[ScienceObservationRecordV1]],
    *,
    next_absolute_tick: int,
) -> dict[str, Any]:
    per_world: dict[str, Any] = {}
    for cell_id in sorted(records_by_cell):
        rows = tuple(records_by_cell[cell_id])
        if not rows:
            raise CampaignProviderError(
                f"typed science world {cell_id} has no canonical observations"
            )
        sequences = tuple(row.world_local_sequence for row in rows)
        if sequences != tuple(range(len(rows))):
            raise CampaignProviderError(
                f"typed science world {cell_id} observation sequence is not contiguous"
            )
        per_world[cell_id] = {
            "cell_id": cell_id,
            "logical_world_id": rows[0].logical_world_id,
            "world_uid": rows[0].world_uid.to_dict(),
            "record_count": len(rows),
            "next_world_local_sequence": len(rows),
            "last_absolute_tick": rows[-1].absolute_tick,
            "prefix_chain_sha256": canonical_science_record_stream_sha256_v1(
                rows
            ).to_dict(),
        }
    return {
        "schema_version": "eiel_cuda_full_evolution_science_observation_frontier_v1",
        "observation_mode": SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
        "observation_abi": SCIENCE_OBSERVATION_ABI_V1,
        "next_absolute_tick": next_absolute_tick,
        "per_world": per_world,
    }


def _science_population_count_v1(
    rows: Sequence[ScienceObservationRecordV1],
) -> int:
    for row in reversed(rows):
        if row.record_kind.value != "world_state_snapshot":
            continue
        payload = row.payload.to_value()
        values = payload.get("qualified_field_values")
        if isinstance(values, Mapping) and type(values.get("active_count")) is int:
            return int(values["active_count"])
    return 0


def _science_sidecar_snapshot_v1(
    *,
    context: Any,
    request: ResidentBatchRequestV1,
    records_by_cell: Mapping[str, Sequence[ScienceObservationRecordV1]],
    next_absolute_tick: int,
    terminal: bool,
) -> dict[str, Any]:
    frontier = _science_frontier_v1(
        records_by_cell, next_absolute_tick=next_absolute_tick
    )
    body = {
        "schema_version": SCIENCE_OBSERVATION_SIDECAR_SCHEMA_V1,
        "observation_abi": SCIENCE_OBSERVATION_ABI_V1,
        "observation_mode": SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
        "batch_id": request.batch_id,
        "cell_ids": list(request.cell_ids),
        "records_by_cell": {
            cell_id: [row.to_dict() for row in records_by_cell[cell_id]]
            for cell_id in sorted(records_by_cell)
        },
        "frontier": frontier,
    }
    root = Path(context.run_root) / "science_observations_v1" / request.batch_id
    suffix = "terminal" if terminal else f"tick-{next_absolute_tick:012d}"
    path = root / f"records-{suffix}.json"
    payload = canonical_bytes(body)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            raise CampaignProviderError(
                f"typed science sidecar snapshot already exists with different content: {path}"
            )
    else:
        _write_bytes_durable(path, payload)
        _seal_read_only_file_v1(path)
    context.record_artifact(
        path,
        artifact_family="cuda_full_evolution_v2.science_observation_sidecar",
        retention_class="R3",
        release_predicate=(
            "engineering observation sidecar retained for reducer/review; no automatic release"
        ),
        replacement_verified=False,
    )
    return {
        **frontier,
        "records_path": str(path),
        "records_sha256": digest,
        "terminal": terminal,
    }


def _load_science_sidecar_v1(
    *,
    context: Any,
    request: ResidentBatchRequestV1,
    frontier: Mapping[str, Any],
) -> dict[str, list[ScienceObservationRecordV1]]:
    path_value = frontier.get("records_path")
    supplied_sha = frontier.get("records_sha256")
    if not isinstance(path_value, str) or not isinstance(supplied_sha, str):
        raise CampaignProviderError("typed science checkpoint lacks sidecar identity")
    supplied_path = Path(path_value).expanduser()
    if supplied_path.is_symlink():
        raise CampaignProviderError("typed science sidecar is a symbolic link")
    path = supplied_path.resolve()
    try:
        path.relative_to(Path(context.run_root).resolve())
    except ValueError as exc:
        raise CampaignProviderError("typed science sidecar escapes its bound run") from exc
    if not path.is_file() or path.is_symlink():
        raise CampaignProviderError("typed science sidecar is unavailable")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != supplied_sha:
        raise CampaignProviderError("typed science sidecar digest differs")
    try:
        body = json.loads(payload)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CampaignProviderError("typed science sidecar is not valid JSON") from exc
    if canonical_bytes(body) != payload:
        raise CampaignProviderError("typed science sidecar bytes are not canonical")
    required = {
        "schema_version",
        "observation_abi",
        "observation_mode",
        "batch_id",
        "cell_ids",
        "records_by_cell",
        "frontier",
    }
    if (
        not isinstance(body, Mapping)
        or set(body) != required
        or body.get("schema_version") != SCIENCE_OBSERVATION_SIDECAR_SCHEMA_V1
        or body.get("observation_abi") != SCIENCE_OBSERVATION_ABI_V1
        or body.get("observation_mode") != SCIENCE_OBSERVATION_TYPED_SIDECAR_V1
        or body.get("batch_id") != request.batch_id
        or body.get("cell_ids") != list(request.cell_ids)
    ):
        raise CampaignProviderError("typed science sidecar contract is invalid")
    raw_rows = body.get("records_by_cell")
    if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(request.cell_ids):
        raise CampaignProviderError("typed science sidecar world set differs")
    records: dict[str, list[ScienceObservationRecordV1]] = {}
    from engine.native_runtime.cuda_v2_science_capture_v1 import (
        science_capture_producer_phase_registry_v1,
    )

    phase_registry = science_capture_producer_phase_registry_v1()
    for cell_id in request.cell_ids:
        values = raw_rows[cell_id]
        if not isinstance(values, list):
            raise CampaignProviderError("typed science sidecar rows are invalid")
        try:
            rows = [ScienceObservationRecordV1.from_dict(row) for row in values]
        except Exception as exc:
            raise CampaignProviderError(
                f"typed science sidecar row validation failed for {cell_id}"
            ) from exc
        if not rows or any(row.cell_id != cell_id for row in rows):
            raise CampaignProviderError(
                f"typed science sidecar record ownership differs for {cell_id}"
            )
        if tuple(row.world_local_sequence for row in rows) != tuple(range(len(rows))):
            raise CampaignProviderError(
                f"typed science sidecar sequence is not contiguous for {cell_id}"
            )
        if any(
            row.instrument_manifest_sha256 != rows[0].instrument_manifest_sha256
            or row.science_semantic_acquisition_sha256
            != rows[0].science_semantic_acquisition_sha256
            or row.world_uid != rows[0].world_uid
            or row.logical_world_id != rows[0].logical_world_id
            for row in rows
        ):
            raise CampaignProviderError(
                f"typed science sidecar record identity changes within {cell_id}"
            )
        try:
            validate_science_observation_stream_v1(
                rows,
                producer_phase_registry_ordinals=phase_registry,
                expected_start_world_local_sequence=0,
            )
        except Exception as exc:
            raise CampaignProviderError(
                f"typed science sidecar canonical stream validation failed for {cell_id}"
            ) from exc
        control_digest = _science_control_binding_digest_v1(
            rows,
            expected_native_control=request.native_control,
        )
        expected_control = ContentDigest(
            str(control_digest["algorithm"]), str(control_digest["hex"])
        )
        if any(row.control_binding_sha256 != expected_control for row in rows):
            raise CampaignProviderError(
                f"typed science sidecar control-binding join differs for {cell_id}"
            )
        records[cell_id] = rows
    observed_frontier = _science_frontier_v1(
        records, next_absolute_tick=int(frontier.get("next_absolute_tick", -1))
    )
    if body.get("frontier") != observed_frontier:
        raise CampaignProviderError("typed science sidecar frontier is not self-consistent")
    expected = dict(frontier)
    actual = {
        **observed_frontier,
        "records_path": str(path),
        "records_sha256": supplied_sha,
        "terminal": bool(frontier.get("terminal", False)),
    }
    if expected != actual:
        raise CampaignProviderError("typed science checkpoint frontier differs from sidecar")
    return records


def _science_append_output_v1(
    *,
    outputs: Mapping[str, Mapping[str, object]],
    records_by_cell: dict[str, list[ScienceObservationRecordV1]],
    cells: Mapping[str, Mapping[str, Any]],
) -> None:
    for cell_id in sorted(outputs):
        cell = cells.get(cell_id)
        if not isinstance(cell, Mapping):
            raise CampaignProviderError(f"typed science output names unknown cell {cell_id}")
        finalized = _science_observation_projection.finalize_native_cuda_v2_observations_v1(
            outputs[cell_id],
            context=_science_context_v1(cell),
            start_world_local_sequence=len(records_by_cell[cell_id]),
        )
        prior = records_by_cell[cell_id]
        controls = [
            row
            for row in finalized
            if row.record_kind.value == "control_binding"
        ]
        if prior:
            if controls:
                raise CampaignProviderError(
                    "typed science world repeated its immutable control-binding row"
                )
            expected_control = prior[0].control_binding_sha256
        else:
            if len(controls) != 1:
                raise CampaignProviderError(
                    "typed science world must begin with exactly one control-binding row"
                )
            expected_control = controls[0].control_binding_sha256
        if any(
            row.control_binding_sha256 != expected_control for row in finalized
        ):
            raise CampaignProviderError(
                f"typed science output control-binding join differs for {cell_id}"
            )
        records_by_cell[cell_id].extend(finalized)


def _science_control_binding_digest_v1(
    rows: Sequence[ScienceObservationRecordV1],
    *,
    expected_native_control: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return the one control digest and verify native provenance on it.

    Checkpoint and completion rows are produced after the initial capture
    envelope.  They must join the original control binding rather than mint a
    marker-local digest.  When the provider has a native control receipt, the
    control row must use the native payload variant and carry the exact receipt
    that was validated at readiness.
    """

    controls = [
        row
        for row in rows
        if row.record_kind.value == "control_binding"
    ]
    if len(controls) != 1:
        raise CampaignProviderError(
            "typed science world must contain exactly one control-binding row"
        )
    control = controls[0]
    if expected_native_control is not None:
        if control.payload.schema_version != (
            SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION
        ):
            raise CampaignProviderError(
                "native science control binding lacks the applied-control payload variant"
            )
        payload = control.payload.to_value()
        if not isinstance(payload, Mapping):
            raise CampaignProviderError("native science control payload is not an object")
        applied = payload.get("applied_native_control")
        if applied != dict(expected_native_control):
            raise CampaignProviderError(
                "native science control binding differs from the applied native receipt"
            )
        try:
            validated_native_control = validate_science_native_control_receipt_v1(
                applied,
                field="control_binding.applied_native_control",
            )
        except Exception as exc:
            raise CampaignProviderError(
                "native science control binding carries invalid applied provenance"
            ) from exc
        if validated_native_control != dict(expected_native_control):
            raise CampaignProviderError(
                "native science control binding differs from the validated native receipt"
            )
        if canonical_sha256(payload) != control.control_binding_sha256.hex:
            raise CampaignProviderError(
                "native science control binding digest does not bind its payload"
            )
    return control.control_binding_sha256.to_dict()


def _tensor_payload_v1(value: Any, *, name: str) -> tuple[bytes, str, list[int]]:
    try:
        tensor = value.detach().contiguous().cpu()
        payload = tensor.numpy().tobytes(order="C")
        return payload, str(tensor.dtype), [int(item) for item in tensor.shape]
    except Exception:
        if isinstance(value, bytes):
            return value, "uint8", [len(value)]
        raise CampaignProviderError(f"cannot persist CUDA event tensor: {name}")


def _persist_event_dispatch_v1(
    runtime: Any,
    dispatch: Any,
    *,
    scratch_root: Path,
    receipt_factory: Callable[..., Any] = seal_event_sink_receipt_v2,
) -> dict[str, Any]:
    start = int(dispatch.start_next_absolute_tick)
    end = int(dispatch.end_next_absolute_tick)
    tick_count = int(dispatch.tick_count)
    if tick_count <= 0 or end != start + tick_count:
        raise CampaignProviderError("CUDA Native V2 event dispatch span is invalid")
    name = f"dispatch-tick-{start:012d}-{end:012d}"
    events_root = scratch_root / "events"
    events_root.mkdir(parents=True, exist_ok=True)
    final = events_root / name
    if final.exists() or final.is_symlink():
        raise CampaignProviderError("CUDA Native V2 event dispatch path already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{name}.", suffix=".tmp", dir=events_root))
    try:
        rows_bytes, rows_dtype, rows_shape = _tensor_payload_v1(
            dispatch.event_rows, name="event_rows"
        )
        counts_bytes, counts_dtype, counts_shape = _tensor_payload_v1(
            dispatch.event_counts, name="event_counts"
        )
        rows_sha256 = hashlib.sha256(rows_bytes).hexdigest()
        counts_sha256 = hashlib.sha256(counts_bytes).hexdigest()
        metadata = {
            "schema_version": EVENT_PERSISTENCE_SCHEMA_V1,
            "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
            "start_next_absolute_tick": start,
            "end_next_absolute_tick": end,
            "tick_count": tick_count,
            "world_uids": [int(value) for value in dispatch.world_uids],
            "event_rows": {
                "file": "event_rows.bin",
                "dtype": rows_dtype,
                "shape": rows_shape,
                "nbytes": len(rows_bytes),
                "sha256": rows_sha256,
            },
            "event_counts": {
                "file": "event_counts.bin",
                "dtype": counts_dtype,
                "shape": counts_shape,
                "nbytes": len(counts_bytes),
                "sha256": counts_sha256,
            },
            "dispatch_content_sha256": str(dispatch.content_sha256),
            "artifact_sha256": str(dispatch.artifact_sha256),
            "manifest_sha256": str(dispatch.manifest_sha256),
        }
        for path, payload in (
            (stage / "event_rows.bin", rows_bytes),
            (stage / "event_counts.bin", counts_bytes),
            (stage / "metadata.json", canonical_bytes(metadata)),
        ):
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(stage)
        os.replace(stage, final)
        _fsync_directory(events_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    receipt = receipt_factory(
        dispatch,
        sink_id=EVENT_SINK_ID_V1,
        sink_receipt_id=name,
    )
    receipt_payload = _jsonable_v1(receipt)
    if not isinstance(receipt_payload, Mapping):
        raise CampaignProviderError("CUDA Native V2 sink receipt is not serializable")
    receipt_path = final / "sink_receipt.json"
    _write_bytes_durable(receipt_path, canonical_bytes(receipt_payload))
    runtime.acknowledge_event_drain(receipt)
    return {
        "path": str(final),
        "start_next_absolute_tick": start,
        "end_next_absolute_tick": end,
        "tick_count": tick_count,
        "dispatch_content_sha256": str(dispatch.content_sha256),
        "sink_receipt_sha256": str(receipt_payload["receipt_sha256"]),
    }


def _persist_direct_sparse_event_dispatch_v3(
    owner: Any,
    dispatch: Any,
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Commit one owner-issued sparse envelope before advancing its frontier."""

    start = int(dispatch.start_next_absolute_tick)
    end = int(dispatch.end_next_absolute_tick)
    tick_count = int(dispatch.tick_count)
    if tick_count <= 0 or end != start + tick_count:
        raise CampaignProviderError("direct-sparse event dispatch span is invalid")
    sample_id = f"dispatch-tick-{start:012d}-{end:012d}"
    dispatch_root = scratch_root / "events" / sample_id
    try:
        dispatch_root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(dispatch_root.parent)
        prepared = owner.prepare_sparse_event_durable_envelope_v3(
            dispatch, sample_id=sample_id
        )
        payload = prepared.payload_for_publication()
        envelope_path = (dispatch_root / "direct-sparse-events-v3.bin").resolve()
        _write_bytes_durable(envelope_path, payload)
        envelope_path.chmod(stat.S_IRUSR)
        descriptor = os.open(envelope_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(dispatch_root)
        durable_envelope = _direct_sparse_runtime.open_verified_prepared_direct_sparse_durable_envelope_v3(
            prepared, durable_path=envelope_path
        )
        try:
            receipt = _direct_sparse_runtime.seal_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                sink_id=DIRECT_SPARSE_EVENT_SINK_ID_V3,
                sink_receipt_id=f"{sample_id}:{prepared.content_sha256}",
                sample_id=sample_id,
                durable_envelope=durable_envelope,
            )
            receipt_payload = _jsonable_v1(receipt)
            if not isinstance(receipt_payload, Mapping):
                raise CampaignProviderError(
                    "direct-sparse sink receipt is not serializable"
                )
            receipt_path = dispatch_root / "sink_receipt.json"
            _write_bytes_durable(receipt_path, canonical_bytes(receipt_payload))
            _seal_read_only_file_v1(receipt_path)
            owner.acknowledge_sparse_event_drain_v3(
                receipt, durable_envelope=durable_envelope
            )
        finally:
            durable_envelope.close()
    except Exception as exc:
        if bool(getattr(owner, "has_pending_dispatch", False)):
            try:
                owner.reject_pending_sparse_event_drain_v3(
                    reason=f"canonical_direct_sparse_publication_failed_{type(exc).__name__}"
                )
            except Exception as rollback_exc:
                raise CampaignProviderError(
                    "direct-sparse durable publication failed and owner rollback "
                    f"did not converge: {type(rollback_exc).__name__}: {rollback_exc}"
                ) from exc
        raise CampaignProviderError(
            f"direct-sparse durable publication failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "schema_version": "eiel_cuda_full_evolution_direct_sparse_persistence_v3",
        "path": str(dispatch_root),
        "durable_envelope_path": str(envelope_path),
        "durable_envelope_bytes": int(prepared.nbytes),
        "durable_envelope_file_sha256": str(prepared.content_sha256),
        "start_next_absolute_tick": start,
        "end_next_absolute_tick": end,
        "tick_count": tick_count,
        "dispatch_content_sha256": str(dispatch.content_sha256),
        "durable_content_sha256": str(prepared.content_sha256),
        "sink_receipt_sha256": str(receipt_payload["receipt_sha256"]),
    }


def _bind_direct_sparse_envelope_to_compact_dispatch_v1(
    event_receipt: dict[str, Any],
    compact_receipt: Mapping[str, Any],
) -> None:
    """Bind compact identity while lifecycle retains the exact R0 source."""

    if (
        event_receipt.get("dispatch_content_sha256")
        != compact_receipt.get("event_content_sha256")
        or event_receipt.get("start_next_absolute_tick")
        != compact_receipt.get("start_next_absolute_tick")
        or event_receipt.get("end_next_absolute_tick")
        != compact_receipt.get("end_next_absolute_tick")
        or compact_receipt.get("dispatch_payload_storage_encoding")
        != COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1
    ):
        raise CampaignProviderError(
            "compact dispatch replacement identity differs from its sparse envelope"
        )
    artifact_sha256 = compact_receipt.get("artifact_sha256")
    event_uncompressed_sha256 = compact_receipt.get("event_uncompressed_sha256")
    source_sink_receipt_sha256 = compact_receipt.get(
        "source_sink_receipt_sha256"
    )
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or not isinstance(event_uncompressed_sha256, str)
        or len(event_uncompressed_sha256) != 64
        or not isinstance(source_sink_receipt_sha256, str)
        or len(source_sink_receipt_sha256) != 64
        or source_sink_receipt_sha256
        != event_receipt.get("sink_receipt_sha256")
    ):
        raise CampaignProviderError(
            "compact dispatch replacement digest or source sink binding is invalid"
        )
    path = Path(str(event_receipt.get("durable_envelope_path") or ""))
    if path.is_symlink() or not path.is_file():
        raise CampaignProviderError("direct-sparse replacement source is absent")
    expected_bytes = event_receipt.get("durable_envelope_bytes")
    expected_sha256 = event_receipt.get("durable_envelope_file_sha256")
    if (
        type(expected_bytes) is not int
        or path.stat().st_size != expected_bytes
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha256_file(path) != expected_sha256
    ):
        raise CampaignProviderError(
            "direct-sparse replacement source identity changed before release"
        )
    event_receipt.update(
        {
            "event_materialization_mode": (
                "raw_r0_retained_with_compact_identity_binding_v1"
            ),
            "durable_envelope_released": False,
            "compact_identity_binding_verified": True,
            "compact_identity_binding": {
                "path": str(compact_receipt.get("path")),
                "artifact_sha256": artifact_sha256,
                "event_content_sha256": compact_receipt[
                    "event_content_sha256"
                ],
                "event_uncompressed_sha256": event_uncompressed_sha256,
                "sink_receipt_sha256": event_receipt["sink_receipt_sha256"],
                "source_sink_receipt_sha256": source_sink_receipt_sha256,
            },
        }
    )


def _settle_compact_science_publication_transaction_v1(
    *,
    pipeline: Any,
    token: Any,
    event_receipt: dict[str, Any],
    direct_sparse: bool,
    context: Any,
    producer_backpressure: bool,
    publish: Callable[..., dict[str, Any]] = (
        _compact_science_capture.publish_frozen_compact_science_dispatch_v1
    ),
    bind_direct_sparse: Callable[[dict[str, Any], Mapping[str, Any]], None] = (
        _bind_direct_sparse_envelope_to_compact_dispatch_v1
    ),
    record_root: Callable[..., Mapping[str, Any]] = (
        _record_compact_science_root_v1
    ),
    reconcile_inventory: Callable[..., str] = (
        _reconcile_compact_science_inventory_commit_v1
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Commit one compact stage or remove every externally visible result."""

    event_receipt_before_settlement = dict(event_receipt)
    published_root: Path | None = None
    inventory_state = "NOT_ATTEMPTED"
    artifact_family = "cuda_full_evolution_v2.compact_science_dispatch"
    try:
        staged_receipt = pipeline.result(
            token,
            producer_backpressure=producer_backpressure,
        )
        compact_receipt = publish(
            staged_receipt,
            destination=token.destination,
        )
        published_root = Path(str(compact_receipt["path"]))
        if direct_sparse:
            bind_direct_sparse(event_receipt, compact_receipt)
        compact_pointer = _compact_science_artifact_pointer_v1(
            compact_receipt
        )
        try:
            record_root(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
        except BaseException as record_error:
            inventory_state = reconcile_inventory(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
            if inventory_state != "COMMITTED_EXACT_DURABLE":
                raise CampaignProviderError(
                    "compact science lifecycle record failed with inventory "
                    f"state {inventory_state}"
                ) from record_error
        else:
            inventory_state = reconcile_inventory(
                context,
                compact_receipt["path"],
                artifact_family=artifact_family,
            )
            if inventory_state != "COMMITTED_EXACT_DURABLE":
                raise CampaignProviderError(
                    "compact science lifecycle record returned without a "
                    "durable exact inventory commit: "
                    f"{inventory_state}"
                )
        pipeline.acknowledge_published(
            token,
            published_receipt=compact_receipt,
        )
        return dict(compact_receipt), compact_pointer
    except BaseException as exc:
        if published_root is not None and inventory_state == "NOT_ATTEMPTED":
            inventory_state = reconcile_inventory(
                context,
                published_root,
                artifact_family=artifact_family,
            )
        preserve_root = inventory_state in {
            "COMMITTED_EXACT_DURABLE",
            "COMMITTED_VISIBLE_NOT_DURABLE_PRESERVE_ROOT",
            "COMMIT_UNKNOWN_PRESERVE_ROOT",
        }
        cleanup_error: BaseException | None = None
        if (
            not preserve_root
            and published_root is not None
            and os.path.lexists(published_root)
        ):
            try:
                _compact_science_capture.remove_compact_science_stage_v1(
                    published_root
                )
            except BaseException as rollback_exc:
                cleanup_error = rollback_exc
        if (
            not preserve_root
            and published_root is not None
            and os.path.lexists(published_root)
        ):
            raise CampaignProviderError(
                "compact science settlement failed and its canonical root "
                "cleanup did not converge: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            ) from exc
        if not preserve_root:
            event_receipt.clear()
            event_receipt.update(event_receipt_before_settlement)
        if isinstance(exc, CompactScienceAsyncPublicationError):
            raise CampaignProviderError(
                f"compact science async publication failed: {exc}"
            ) from exc
        if isinstance(exc, Exception):
            raise CampaignProviderError(
                "compact science async publication settlement failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise


def _checkpoint_runtime_body_v1(
    checkpoint: CudaFullEvolutionCheckpointV2,
) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.schema_version,
        "time_authority": checkpoint.time_authority,
        "contract": checkpoint.contract.to_dict(),
        "canonical_state_shape": list(checkpoint.canonical_state_shape),
        "canonical_state_nbytes": checkpoint.canonical_state_nbytes,
        "canonical_state_sha256": checkpoint.canonical_state_sha256,
        "canonical_state_layout_sha256": checkpoint.canonical_state_layout_sha256,
        "initial_next_absolute_tick": checkpoint.initial_next_absolute_tick,
        "advanced_ticks": checkpoint.advanced_ticks,
        "acknowledged_event_next_tick": checkpoint.acknowledged_event_next_tick,
        "last_sink_receipt_sha256": checkpoint.last_sink_receipt_sha256,
        "world_uids": list(checkpoint.world_uids),
        "rng_keys": list(checkpoint.rng_keys),
        "rng_semantic_epoch": checkpoint.rng_semantic_epoch,
        "artifact_sha256": checkpoint.artifact_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "device_identity": dict(checkpoint.device_identity),
        "population_fixture_id": checkpoint.population_fixture_id,
    }


def _validated_checkpoint_runtime_projection_v1(
    checkpoint: CudaFullEvolutionCheckpointV2,
    *,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one native checkpoint once for synchronous or ACS publication."""

    try:
        _cuda_v2_runtime._validate_checkpoint_envelope(checkpoint)
    except Exception as exc:
        raise CampaignProviderError(
            f"CUDA Native V2 checkpoint envelope is invalid: {exc}"
        ) from exc
    runtime_body = _checkpoint_runtime_body_v1(checkpoint)
    if canonical_sha256(runtime_body) != checkpoint.metadata_seal_sha256:
        raise CampaignProviderError("CUDA Native V2 runtime metadata seal mismatch")
    if (
        checkpoint.artifact_sha256 != artifact["binary_sha256"]
        or checkpoint.manifest_sha256 != artifact["manifest_sha256"]
        or Path(checkpoint.artifact_path).resolve()
        != Path(str(artifact["binary_path"])).resolve()
    ):
        raise CampaignProviderError("checkpoint binds another artifact unit")
    return {
        **runtime_body,
        "metadata_seal_sha256": checkpoint.metadata_seal_sha256,
    }


def _publish_checkpoint_bundle_v1(
    checkpoint: CudaFullEvolutionCheckpointV2,
    *,
    context: Any,
    artifact: Mapping[str, Any],
    occurrence_adapter: AbsoluteTickCheckpointOccurrenceAdapterV1,
) -> tuple[Path, str, dict[str, str], str]:
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_V2:
        raise CampaignProviderError("CUDA Native V2 checkpoint schema mismatch")
    tick = int(checkpoint.acknowledged_event_next_tick)
    checkpoint_id = (
        f"cuda-native-checkpoint-tick-{tick:012d}-"
        f"{checkpoint.canonical_state_sha256[:16]}"
    )
    root = Path(context.batch_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint-tick-{tick:012d}"
    if final.exists() or final.is_symlink():
        raise CampaignProviderError("CUDA Native V2 checkpoint path already exists")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".checkpoint-tick-{tick:012d}.", suffix=".tmp", dir=root
        )
    )
    try:
        runtime_checkpoint = _validated_checkpoint_runtime_projection_v1(
            checkpoint,
            artifact=artifact,
        )
        metadata = {
            "schema_version": CHECKPOINT_METADATA_SCHEMA_V1,
            "checkpoint_format_id": CHECKPOINT_SCHEMA_V2,
            "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
            "checkpoint_id": str(checkpoint_id),
            "absolute_next_tick": tick,
            "artifact_path": str(Path(checkpoint.artifact_path).resolve()),
            "runtime_checkpoint": runtime_checkpoint,
        }
        metadata_bytes = canonical_bytes(metadata)
        completion = {
            "schema_version": CHECKPOINT_COMPLETION_SCHEMA_V1,
            "status": "complete",
            "checkpoint_format_id": CHECKPOINT_SCHEMA_V2,
            "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
            "checkpoint_id": str(checkpoint_id),
            "absolute_next_tick": tick,
            "canonical_state_file": "canonical_state.bin",
            "canonical_state_nbytes": checkpoint.canonical_state_nbytes,
            "canonical_state_sha256": checkpoint.canonical_state_sha256,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }
        for path, payload in (
            (stage / "canonical_state.bin", checkpoint.canonical_state),
            (stage / "metadata.json", metadata_bytes),
            (stage / "bundle_complete.json", canonical_bytes(completion)),
        ):
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(stage)
        os.replace(stage, final)
        _fsync_directory(root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    try:
        tree_digest = _seal_published_checkpoint_bundle_v1(
            final,
            checkpoint_id=checkpoint_id,
            absolute_next_tick=tick,
            context=context,
            occurrence_adapter=occurrence_adapter,
        )
    except BaseException as exc:
        failed = _quarantine_provider_publication_v1(final)
        if failed is not None:
            raise CampaignProviderError(
                "CUDA Native V2 checkpoint bundle was quarantined after its "
                f"external seal failed: {failed}"
            ) from exc
        raise
    return final, checkpoint_id, tree_digest, checkpoint.canonical_state_sha256


def _quarantine_provider_publication_v1(path: Path) -> Path | None:
    """Move one exact provider-owned publication out of its canonical name."""

    if path.is_symlink():
        raise CampaignProviderError(
            "CUDA Native V2 failed publication path is a symbolic link"
        )
    if not path.exists():
        return None
    failed = path.with_name(f"{path.name}.publication.failed.{uuid4().hex}")
    while failed.exists() or failed.is_symlink():
        failed = path.with_name(f"{path.name}.publication.failed.{uuid4().hex}")
    os.replace(path, failed)
    _fsync_directory(path.parent)
    return failed


def _quarantine_after_external_commit_failure_v1(
    path: Path | None,
    failure: BaseException,
    *,
    label: str,
) -> None:
    if path is None:
        return
    try:
        failed = _quarantine_provider_publication_v1(path)
    except BaseException as cleanup_error:
        failure.add_note(
            f"CUDA Native V2 {label} quarantine also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    else:
        if failed is not None:
            failure.add_note(f"CUDA Native V2 {label} quarantined at {failed}")


def _seal_published_checkpoint_bundle_v1(
    bundle: Path,
    *,
    checkpoint_id: str,
    absolute_next_tick: int,
    context: Any,
    occurrence_adapter: AbsoluteTickCheckpointOccurrenceAdapterV1,
) -> dict[str, str]:
    """Seal an already-published native bundle through the CRCA seam."""

    sealed = occurrence_adapter.seal_bundle_v1(
        bundle,
        checkpoint_id=checkpoint_id,
        absolute_next_tick=absolute_next_tick,
        owner_id="engine.campaigns.cuda_full_evolution_v2_provider",
        owner_digest_sha256=str(context.execution_route_sha256),
    )
    if not isinstance(sealed, Mapping):
        raise CampaignProviderError("checkpoint occurrence adapter returned no seal")
    return _digest_payload_v1(
        sealed.get("checkpoint_bundle_digest"),
        name="checkpoint bundle digest",
    )


def _digest_payload_v1(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise CampaignProviderError(f"{name} must be an exact digest object")
    if value.get("algorithm") != "sha256":
        raise CampaignProviderError(f"{name} must use SHA-256")
    return {
        "algorithm": "sha256",
        "hex": _sha256_text(value.get("hex"), name=name),
    }


def _authoritative_attempt_identity_v1(context: Any) -> Path | None:
    """Locate, but never interpret, the Scheduler-owned authority bundle."""

    path = Path(context.run_root) / ATTEMPT_AUTHORITY_NAME
    if not path.exists() and not path.is_symlink():
        return None
    return _regular_unlinked_file(path, name="CRCA attempt authority")


def _materialize_authoritative_occurrence_v1(
    bundle: Path,
    *,
    checkpoint_id: str,
    tree_digest: Mapping[str, str],
    absolute_next_tick: int,
    context: Any,
    source_batch_id: str,
    attempt_authority: Path | None,
    parent_checkpoint_digest: Mapping[str, str] | None,
    occurrence_adapter: AbsoluteTickCheckpointOccurrenceAdapterV1,
) -> tuple[dict[str, str] | None, Path | None]:
    if attempt_authority is None:
        return None, None
    occurrence_root = (
        Path(context.run_root)
        / "control"
        / "checkpoint-occurrences"
        / source_batch_id
        / bundle.name
    )
    try:
        result = occurrence_adapter.materialize_occurrence_v1(
            bundle,
            occurrence_root,
            attempt_authority_path=attempt_authority,
            checkpoint_id=checkpoint_id,
            absolute_next_tick=absolute_next_tick,
            owner_id="engine.campaigns.cuda_full_evolution_v2_provider",
            owner_digest_sha256=str(context.execution_route_sha256),
            source_batch_id=source_batch_id,
            parent_checkpoint_digest=parent_checkpoint_digest,
        )
        if not isinstance(result, Mapping):
            raise CampaignProviderError(
                "checkpoint occurrence adapter returned no materialization"
            )
        observed_tree = _digest_payload_v1(
            result.get("checkpoint_bundle_digest"),
            name="materialized checkpoint bundle digest",
        )
        if observed_tree != dict(tree_digest):
            raise CampaignProviderError(
                "checkpoint occurrence adapter bound another checkpoint tree"
            )
        occurrence_digest = _digest_payload_v1(
            result.get("occurrence_digest"), name="checkpoint occurrence digest"
        )
    except BaseException as exc:
        failed = _quarantine_provider_publication_v1(occurrence_root)
        if failed is not None:
            raise CampaignProviderError(
                "CUDA Native V2 checkpoint occurrence was quarantined after "
                f"external commit failed: {failed}"
            ) from exc
        raise
    return occurrence_digest, occurrence_root


def _checkpoint_record_v1(
    *,
    request: ResidentBatchRequestV1,
    context: Any,
    bundle: Path,
    checkpoint_id: str,
    tree_digest: Mapping[str, str],
    state_sha256: str,
    absolute_next_tick: int,
    occurrence_digest: Mapping[str, str] | None,
    occurrence_root: Path | None,
    completed_restore_ticks: Sequence[int],
    science_continuation_admission: Mapping[str, Any],
    science_observation_frontier: Mapping[str, Any] | None = None,
    compact_science_frontier: Mapping[str, Any] | None = None,
    native_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity_sha256 = (
        str(occurrence_digest["hex"])
        if occurrence_digest is not None
        else canonical_sha256(
            {
                "checkpoint_id": str(checkpoint_id),
                "absolute_next_tick": absolute_next_tick,
                "checkpoint_bundle_digest": dict(tree_digest),
                "execution_route_sha256": context.execution_route_sha256,
            }
        )
    )
    result = {
        "schema_version": PROVIDER_CHECKPOINT_SCHEMA_V1,
        "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
        "checkpoint_id": str(checkpoint_id),
        "source_batch_id": request.batch_id,
        "ordered_cell_ids": list(request.cell_ids),
        "resident_request_sha256": canonical_sha256(request.payload()),
        "next_absolute_tick": absolute_next_tick,
        "checkpoint_at_tick": absolute_next_tick,
        "absolute_tick": absolute_next_tick,
        "next_tick": absolute_next_tick,
        "checkpoint_identity_sha256": identity_sha256,
        "checkpoint_bundle_digest": dict(tree_digest),
        "state_sha256": state_sha256,
        "path": str(bundle),
        "occurrence_status": (
            "CANONICAL_CRCA_VISIBLE"
            if occurrence_digest is not None
            else "NOT_MATERIALIZED_MISSING_ATTEMPT_AUTHORITY"
        ),
        "occurrence_root": None if occurrence_root is None else str(occurrence_root),
        "occurrence_digest": (
            None if occurrence_digest is None else dict(occurrence_digest)
        ),
        "deliberate_restore_ticks_completed": list(completed_restore_ticks),
        "science_continuation_admission": dict(science_continuation_admission),
        "cell_bundles": [
            {
                "path": str(bundle),
                "artifact_family": CHECKPOINT_ARTIFACT_FAMILY_V1,
                "cell_ids": list(request.cell_ids),
            }
        ],
        "execution_route_sha256": str(context.execution_route_sha256),
    }
    if native_control is not None:
        result["native_control"] = dict(native_control)
    if science_observation_frontier is not None:
        result["science_observation_frontier"] = dict(science_observation_frontier)
    if compact_science_frontier is not None:
        result["compact_science_frontier"] = dict(compact_science_frontier)
    return result


def _restore_checkpoint_bundle_v1(
    checkpoint: Mapping[str, Any],
    *,
    context: Any,
    artifact: Mapping[str, Any],
    request: ResidentBatchRequestV1,
    attempt_authority: Path | None,
    occurrence_adapter: AbsoluteTickCheckpointOccurrenceAdapterV1,
    expected_native_control: Mapping[str, Any] | None = None,
) -> CudaFullEvolutionCheckpointV2:
    if checkpoint.get("schema_version") != PROVIDER_CHECKPOINT_SCHEMA_V1:
        raise CampaignProviderError("unsupported CUDA Native V2 provider checkpoint")
    if checkpoint.get("time_authority") != CUDA_NATIVE_TIME_AUTHORITY_V1:
        raise CampaignProviderError("checkpoint is not absolute-tick authoritative")
    source_batch_id = request.batch_id
    if (
        checkpoint.get("source_batch_id") != source_batch_id
        or checkpoint.get("ordered_cell_ids") != list(request.cell_ids)
        or checkpoint.get("resident_request_sha256")
        != canonical_sha256(request.payload())
    ):
        raise CampaignProviderError("checkpoint binds another physical batch")
    if checkpoint.get("science_continuation_admission") != (
        _base_checkpoint_science_admission_v1(artifact)
    ):
        raise CampaignProviderError(
            "checkpoint lacks the exact fail-closed science-continuation admission"
        )
    if expected_native_control is not None and checkpoint.get("native_control") != dict(
        expected_native_control
    ):
        raise CampaignProviderError(
            "checkpoint native-control binding differs from the requested batch"
        )
    tick = checkpoint.get("next_absolute_tick")
    if type(tick) is not int or any(
        checkpoint.get(name) != tick
        for name in ("checkpoint_at_tick", "absolute_tick", "next_tick")
    ):
        raise CampaignProviderError("checkpoint absolute-tick aliases conflict")
    if attempt_authority is None:
        raise CampaignProviderError(
            "authoritative continuation requires control/crca/attempt-authority.json"
        )
    occurrence_root = checkpoint.get("occurrence_root")
    occurrence_digest = checkpoint.get("occurrence_digest")
    if not isinstance(occurrence_root, str) or occurrence_digest is None:
        raise CampaignProviderError("checkpoint lacks an authoritative CRCA occurrence")
    bundle = Path(str(checkpoint.get("path", ""))).expanduser().resolve()
    expected_bundle_parent = (Path(context.batch_root) / "checkpoints").resolve()
    if bundle.parent != expected_bundle_parent:
        raise CampaignProviderError("checkpoint path escapes its bound physical batch")
    expected_occurrence_root = (
        Path(context.run_root)
        / "control"
        / "checkpoint-occurrences"
        / source_batch_id
        / bundle.name
    ).resolve()
    if Path(occurrence_root).expanduser().resolve() != expected_occurrence_root:
        raise CampaignProviderError("checkpoint occurrence path escapes its bound run")
    payload = occurrence_adapter.restore_occurrence_v1(
        bundle,
        expected_occurrence_root,
        attempt_authority_path=attempt_authority,
        source_batch_id=source_batch_id,
        expected_absolute_next_tick=tick,
        expected_occurrence_digest=_digest_payload_v1(
            occurrence_digest, name="occurrence_digest"
        ),
    )
    if not isinstance(payload, Mapping):
        raise CampaignProviderError("checkpoint occurrence adapter returned no restore")
    canonical_state = payload.get("canonical_state")
    if not isinstance(canonical_state, bytes):
        raise CampaignProviderError("restored canonical checkpoint state is not bytes")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CampaignProviderError("restored native checkpoint metadata is absent")
    runtime = metadata.get("runtime_checkpoint")
    if not isinstance(runtime, Mapping):
        raise CampaignProviderError("restored runtime checkpoint metadata is absent")
    if (
        Path(str(metadata.get("artifact_path", ""))).resolve()
        != Path(str(artifact["binary_path"])).resolve()
    ):
        raise CampaignProviderError("restored checkpoint artifact locator differs")
    return CudaFullEvolutionCheckpointV2(
        contract=CudaFullEvolutionNativeContractV2.from_mapping(runtime["contract"]),
        canonical_state=canonical_state,
        canonical_state_shape=tuple(runtime["canonical_state_shape"]),
        canonical_state_nbytes=int(runtime["canonical_state_nbytes"]),
        canonical_state_sha256=str(runtime["canonical_state_sha256"]),
        canonical_state_layout_sha256=str(runtime["canonical_state_layout_sha256"]),
        initial_next_absolute_tick=int(runtime["initial_next_absolute_tick"]),
        advanced_ticks=int(runtime["advanced_ticks"]),
        acknowledged_event_next_tick=int(runtime["acknowledged_event_next_tick"]),
        last_sink_receipt_sha256=runtime["last_sink_receipt_sha256"],
        world_uids=tuple(runtime["world_uids"]),
        rng_keys=tuple(runtime["rng_keys"]),
        rng_semantic_epoch=int(runtime["rng_semantic_epoch"]),
        artifact_path=Path(str(metadata["artifact_path"])),
        artifact_sha256=str(runtime["artifact_sha256"]),
        manifest_sha256=str(runtime["manifest_sha256"]),
        device_identity=dict(runtime["device_identity"]),
        population_fixture_id=runtime["population_fixture_id"],
        metadata_seal_sha256=str(runtime["metadata_seal_sha256"]),
        schema_version=str(runtime["schema_version"]),
        time_authority=str(runtime["time_authority"]),
    )


def _seed_and_world_uids_v1(
    request: ResidentBatchRequestV1,
) -> tuple[int, tuple[int, ...]]:
    seeds = dict(request.seed_arrays)
    ecology = seeds.get("ecology")
    task = seeds.get("task")
    explicit_native_uids = request.native_rng_world_uids
    if request.native_world_controls:
        if (
            len(request.native_world_controls) != len(request.cell_ids)
            or len(explicit_native_uids) != len(request.cell_ids)
            or len(set(explicit_native_uids)) != len(explicit_native_uids)
            or any(
                type(value) is not int or value <= 0
                for value in explicit_native_uids
            )
        ):
            raise CampaignProviderError(
                "generic science logical-world runtime identities are invalid"
            )
        if (
            not isinstance(ecology, tuple)
            or len(ecology) != len(request.cell_ids)
            or any(type(value) is not int or value < 0 for value in ecology)
            or not isinstance(task, tuple)
            or len(task) != len(request.cell_ids)
            or any(type(value) is not int or value < 0 for value in task)
        ):
            raise CampaignProviderError(
                "generic science seed arrays are incomplete"
            )
        # The native generic initializer and transition consume every resolved
        # domain seed from the per-world table.  This scalar is deliberately
        # inert on that route and remains only for the legacy owner signature.
        return 0, tuple(int(value) for value in explicit_native_uids)
    if explicit_native_uids:
        if (
            len(explicit_native_uids) != len(request.cell_ids)
            or len(set(explicit_native_uids)) != len(explicit_native_uids)
            or any(
                type(value) is not int or value <= 0 for value in explicit_native_uids
            )
        ):
            raise CampaignProviderError(
                "explicit native RNG world identities are invalid"
            )
        if task != tuple(explicit_native_uids):
            raise CampaignProviderError(
                "task seed identities differ from explicit native RNG identities"
            )
        task = tuple(explicit_native_uids)
    if (
        not isinstance(ecology, tuple)
        or len(ecology) != len(request.cell_ids)
        or len(set(ecology)) != 1
        or not isinstance(task, tuple)
        or len(task) != len(request.cell_ids)
        or len(set(task)) != len(task)
        or any(type(value) is not int or value <= 0 for value in task)
    ):
        raise CampaignProviderError(
            "CUDA Native V2 requires one ecology seed and unique positive task-seed world identities"
        )
    seed = int(ecology[0])
    if seed < 0:
        raise CampaignProviderError("CUDA Native V2 ecology seed must be nonnegative")
    return seed, tuple(int(value) for value in task)


def _execute_native_batch_v1(
    *,
    provider: CudaFullEvolutionV2ExecutionProviderV1,
    plan: Mapping[str, Any],
    request: ResidentBatchRequestV1,
    context: Any,
    resume_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reasons = provider.readiness_reasons(plan, request)
    if reasons:
        raise CampaignProviderError(reasons[0].message)
    occurrence_adapter = provider._checkpoint_occurrence_adapter
    if occurrence_adapter is None:
        raise CampaignProviderError(
            "CUDA Native V2 execution requires an injected canonical "
            "absolute-tick checkpoint occurrence adapter"
        )
    artifact = _artifact_unit_v1(plan, validator=provider._artifact_validator)
    selection = artifact["selection"]
    device = f"cuda:{selection['device_identity']['index']}"
    try:
        loaded = provider._artifact_loader(artifact["binary_path"], device=device)
    except Exception as exc:
        raise CampaignProviderError(
            f"CUDA Native V2 artifact load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        str(getattr(loaded, "artifact_sha256", "")) != artifact["binary_sha256"]
        or str(getattr(loaded, "manifest_sha256", "")) != artifact["manifest_sha256"]
        or dict(getattr(loaded, "device_identity", {})) != selection["device_identity"]
    ):
        raise CampaignProviderError(
            "loaded artifact/device identity differs from sealed selection"
        )
    contract = _native_contract_v1(request)
    execution_mode = str(_uniform_value_v1(request, "cuda_native.execution_mode"))
    event_transport = str(_uniform_value_v1(request, "cuda_native.event_transport"))
    science_observation_mode = str(
        _uniform_value_v1(request, "cuda_native.science_observation_mode")
    )
    science_capture_enabled = (
        science_observation_mode == SCIENCE_OBSERVATION_TYPED_SIDECAR_V1
    )
    compact_science_capture_enabled = (
        science_observation_mode == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
    )
    science_cells: dict[str, Mapping[str, Any]] = (
        _science_capture_cells_v1(plan, request)
        if science_capture_enabled or compact_science_capture_enabled
        else {}
    )
    science_records_by_cell: dict[str, list[ScienceObservationRecordV1]] = {
        cell_id: [] for cell_id in science_cells
    }
    science_state_sha_by_cell: dict[str, dict[str, str]] = {}
    science_checkpoint_occurrences: list[str] = []
    direct_sparse = event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
    native_control = (
        None if request.native_control is None else dict(request.native_control)
    )
    native_world_controls = tuple(
        dict(value) for value in request.native_world_controls
    )
    native_world_control_set_sha256 = (
        _native_world_control_set_hex_v1(plan, request)
        if native_world_controls
        else None
    )
    compact_snapshot_ticks = (
        _science_snapshot_ticks_v1(plan, request)
        if compact_science_capture_enabled
        else ()
    )
    try:
        compact_snapshot_implementation = (
            provider._native_snapshot_implementation_selector()
            if compact_science_capture_enabled
            else "python_v1"
        )
    except Exception as exc:
        raise CampaignProviderError(
            "compact science snapshot implementation selection failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if compact_snapshot_implementation not in _native_snapshot_publication.NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1:
        raise CampaignProviderError(
            f"{COMPACT_SCIENCE_SNAPSHOT_IMPLEMENTATION_ENV_V1} must select "
            "exactly python_v1, native_v1, or native_capsule_v1"
        )
    if compact_science_capture_enabled:
        _validate_native_snapshot_resource_admission_v1(
            plan,
            contract,
            implementation=compact_snapshot_implementation,
            slot_count=provider._native_snapshot_slots,
        )
    compact_dispatch_receipts: list[dict[str, Any]] = []
    compact_snapshot_receipts: list[dict[str, Any]] = []
    compact_snapshot_timing_enabled = _compact_snapshot_timing_enabled_v1()
    compact_snapshot_timing_samples: list[dict[str, Any]] = []
    compact_checkpoint_frontier: dict[str, Any] | None = None
    compact_capture_index: dict[str, Any] | None = None
    capability_mask = (
        int(native_control["capability_mask"])
        if native_control is not None
        else _cuda_v2_runtime.FULL_CAPABILITY_MASK_V2
    )
    native_control_digest = (
        str(native_control["control_digest"]["hex"])
        if native_control is not None
        else None
    )
    native_control_vector = (
        dict(native_control["control_vector"])
        if native_control is not None
        else None
    )
    initial_capacity_rows = selection["selection"].get(
        "initial_capacity_rows_per_world"
    )
    attempt_authority = _authoritative_attempt_identity_v1(context)
    expected_batch_root = (
        Path(context.run_root) / "batches" / request.batch_id
    ).resolve()
    if Path(context.batch_root).resolve() != expected_batch_root:
        raise CampaignProviderError("provider context binds another physical batch")
    plan_resource_estimate = plan.get("resource_estimate")
    try:
        progressive_storage_window = (
            progressive_storage_reservation_window_bytes_v1(
                plan_resource_estimate
                if isinstance(plan_resource_estimate, Mapping)
                else None
            )
        )
    except (ArtifactLifecycleError, ValueError) as exc:
        raise CampaignProviderError(
            f"progressive storage reservation is invalid: {exc}"
        ) from exc
    progressive_storage_guard: RunStorageGuard | None = None
    if progressive_storage_window is not None:
        disk_upper_bound = (
            plan_resource_estimate.get("disk_bytes_upper_bound")
            if isinstance(plan_resource_estimate, Mapping)
            else None
        )
        if (
            not direct_sparse
            or not compact_science_capture_enabled
            or not native_world_controls
            or request.acs_checkpoint_slots != 0
            or type(disk_upper_bound) is not int
            or disk_upper_bound <= 0
        ):
            raise CampaignProviderError(
                "progressive storage reservation is not bound to a supported "
                "generic compact native-world route"
            )
        progressive_storage_guard = getattr(context, "storage_guard", None)
        if not isinstance(progressive_storage_guard, RunStorageGuard):
            raise CampaignProviderError(
                "progressive storage requires the canonical campaign-owned guard"
            )
        if (
            progressive_storage_guard.run_root != Path(context.run_root).resolve()
            or progressive_storage_guard.contract.get("contract_sha256")
            != context.lifecycle_contract.get("contract_sha256")
            or progressive_storage_guard.planned_bytes != disk_upper_bound
            or progressive_storage_guard.reservation_window_bytes
            != progressive_storage_window
        ):
            raise CampaignProviderError(
                "campaign-owned progressive storage guard binding differs"
            )
    physical_batch_authorization: Mapping[str, Any] | None = None
    if attempt_authority is not None:
        authorize_batch = getattr(
            occurrence_adapter, "authorize_physical_batch_member_v1", None
        )
        if not callable(authorize_batch):
            raise CampaignProviderError(
                "canonical occurrence adapter cannot authorize the physical batch"
            )
        physical_batch_authorization = authorize_batch(
            attempt_authority_path=attempt_authority,
            source_batch_id=request.batch_id,
            resident_request=request.payload(),
        )
        if not isinstance(physical_batch_authorization, Mapping):
            raise CampaignProviderError("physical batch authorization is absent")
    acs_pipeline: Any | None = None
    acs_capture_pool: Any | None = None
    acs_capture_mode: str | None = None
    acs_checkpoint_receipts: list[dict[str, Any]] = []
    pending_acs_checkpoints: deque[dict[str, Any]] = deque()
    compact_publication_pipeline: CompactScienceAsyncPublicationV1 | None = None
    pending_compact_event_receipts: deque[dict[str, Any]] = deque()
    compact_publication_telemetry: dict[str, Any] | None = None
    native_snapshot_pipeline: Any | None = None
    compiled_snapshot_plan: Any | None = None
    pending_native_snapshots: deque[dict[str, Any]] = deque()
    native_snapshot_publication_telemetry: dict[str, Any] | None = None
    native_snapshot_encoder_identity: dict[str, Any] | None = None
    native_snapshot_recovered_existing_count = 0
    compact_phase_started = perf_counter()
    compact_phase_timing = {
        "native_dispatch_and_capture_seconds": 0.0,
        "direct_sparse_durable_publication_seconds": 0.0,
        "direct_sparse_lifecycle_inventory_seconds": 0.0,
        "compact_host_freeze_and_submit_seconds": 0.0,
        "compact_fifo_settlement_seconds": 0.0,
        "compact_snapshot_seconds": 0.0,
        "compact_checkpoint_frontier_seconds": 0.0,
    }
    runtime: Any | None = None
    direct_sparse_execution_receipt: Mapping[str, Any] | None = None
    parent_occurrence_digest: dict[str, str] | None = None
    completed_restore_ticks: list[int] = []
    if resume_checkpoint is None:
        seed, world_uids = _seed_and_world_uids_v1(request)
        if direct_sparse:
            runtime = provider._direct_sparse_owner_initializer(
                contract,
                seed=seed,
                artifact_path=artifact["binary_path"],
                device=device,
                world_uids=world_uids,
                initial_capacity_rows=int(initial_capacity_rows),
                capability_mask=capability_mask,
                native_control_digest=native_control_digest,
                native_control_vector=native_control_vector,
                native_world_controls=native_world_controls,
                compact_science_capture=compact_science_capture_enabled,
            )
        else:
            runtime = provider._runtime_initializer(
                contract,
                seed=seed,
                next_absolute_tick=contract.start_next_tick,
                artifact_path=artifact["binary_path"],
                device=device,
                enable_reference=False,
                world_uids=world_uids,
                execution_mode=execution_mode,
            )
        current = contract.start_next_tick
    else:
        restored = _restore_checkpoint_bundle_v1(
            resume_checkpoint,
            context=context,
            artifact=artifact,
            request=request,
            attempt_authority=attempt_authority,
            occurrence_adapter=occurrence_adapter,
            expected_native_control=native_control,
        )
        if restored.contract != contract:
            raise CampaignProviderError("restored checkpoint contract differs")
        _expected_seed, expected_world_uids = _seed_and_world_uids_v1(request)
        if restored.world_uids != expected_world_uids:
            raise CampaignProviderError(
                "restored checkpoint world identities differ from the physical batch"
            )
        if direct_sparse:
            compact_restore_metrics = None
            if compact_science_capture_enabled:
                raw_compact_continuation = resume_checkpoint.get(
                    "compact_science_frontier"
                )
                if not isinstance(raw_compact_continuation, Mapping):
                    raise CampaignProviderError(
                        "compact science continuation lacks its counter frontier"
                    )
                frontier_copy = dict(raw_compact_continuation)
                supplied_frontier_digest = frontier_copy.pop(
                    "frontier_sha256", None
                )
                if (
                    raw_compact_continuation.get("schema_version")
                    != "eiel_cuda_v2_compact_science_continuation_frontier_v1"
                    or raw_compact_continuation.get("next_absolute_tick")
                    != resume_checkpoint.get("next_absolute_tick")
                    or raw_compact_continuation.get(
                        "native_world_control_set_sha256"
                    )
                    != native_world_control_set_sha256
                    or supplied_frontier_digest != canonical_sha256(frontier_copy)
                ):
                    raise CampaignProviderError(
                        "compact science continuation frontier identity differs"
                    )
                raw_compact_frontier = raw_compact_continuation.get(
                    "counter_frontier"
                )
                prior_dispatches = raw_compact_continuation.get(
                    "dispatch_artifacts"
                )
                prior_snapshots = raw_compact_continuation.get(
                    "snapshot_artifacts"
                )
                if (
                    not isinstance(raw_compact_frontier, Mapping)
                    or not isinstance(prior_dispatches, list)
                    or not isinstance(prior_snapshots, list)
                    or any(not isinstance(item, Mapping) for item in prior_dispatches)
                    or any(not isinstance(item, Mapping) for item in prior_snapshots)
                ):
                    raise CampaignProviderError(
                        "compact science continuation artifact ownership is invalid"
                    )
                try:
                    loaded_frontier = (
                        _compact_science_capture.load_compact_science_checkpoint_frontier_v1(
                        raw_compact_frontier,
                        expected_root=context.run_root,
                        expected_absolute_tick=int(
                            resume_checkpoint["next_absolute_tick"]
                        ),
                        expected_world_uids=expected_world_uids,
                        expected_native_world_control_set_sha256=str(
                            native_world_control_set_sha256
                        ),
                        )
                    )
                except Exception as exc:
                    raise CampaignProviderError(
                        f"compact science counter frontier restore failed: {exc}"
                    ) from exc
                compact_restore_metrics = loaded_frontier["tensor"]
                compact_checkpoint_frontier = dict(raw_compact_frontier)
                compact_dispatch_receipts.extend(
                    dict(item) for item in prior_dispatches
                )
                compact_snapshot_receipts.extend(
                    dict(item) for item in prior_snapshots
                )
            runtime = provider._direct_sparse_owner_restorer(
                restored,
                artifact_path=artifact["binary_path"],
                device=device,
                initial_capacity_rows=int(initial_capacity_rows),
                capability_mask=capability_mask,
                native_control_digest=native_control_digest,
                native_control_vector=native_control_vector,
                native_world_controls=native_world_controls,
                compact_science_capture=compact_science_capture_enabled,
                compact_science_agent_metrics=compact_restore_metrics,
            )
        else:
            runtime = provider._runtime_restorer(
                restored,
                artifact_path=artifact["binary_path"],
                device=device,
                enable_reference=False,
                execution_mode=execution_mode,
            )
        current = int(resume_checkpoint["next_absolute_tick"])
        if science_capture_enabled:
            raw_frontier = resume_checkpoint.get("science_observation_frontier")
            if not isinstance(raw_frontier, Mapping):
                raise CampaignProviderError(
                    "typed science continuation lacks its observation frontier"
                )
            science_records_by_cell = _load_science_sidecar_v1(
                context=context,
                request=request,
                frontier=raw_frontier,
            )
            if int(raw_frontier.get("next_absolute_tick", -1)) != current:
                raise CampaignProviderError(
                    "typed science observation frontier differs from restore tick"
                )
            for cell_id, rows in science_records_by_cell.items():
                if rows:
                    latest = rows[-1]
                    if latest.source_state_after_sha256 is not None:
                        science_state_sha_by_cell[cell_id] = (
                            latest.source_state_after_sha256.to_dict()
                        )
                    for row in rows:
                        if row.record_kind.value == "checkpoint_marker":
                            payload = row.payload.to_value()
                            occurrence = payload.get("checkpoint_occurrence_id")
                            if isinstance(occurrence, str):
                                science_checkpoint_occurrences.append(occurrence)
        raw_completed = resume_checkpoint.get("deliberate_restore_ticks_completed", [])
        if not isinstance(raw_completed, list) or any(
            type(value) is not int for value in raw_completed
        ):
            raise CampaignProviderError("checkpoint restore frontier is invalid")
        completed_restore_ticks = sorted(set(int(value) for value in raw_completed))
        if current in request.deliberate_restore_ticks:
            completed_restore_ticks = sorted({*completed_restore_ticks, current})
        parent_occurrence_digest = _digest_payload_v1(
            resume_checkpoint["occurrence_digest"], name="parent occurrence digest"
        )
    try:
        if compact_science_capture_enabled:
            runtime_binding = getattr(
                runtime, "native_world_control_binding_v1", None
            )
            if (
                not isinstance(runtime_binding, Mapping)
                or runtime_binding.get("control_set_sha256")
                != native_world_control_set_sha256
            ):
                raise CampaignProviderError(
                    "native owner compact science control-set binding differs"
                )
        if (
            compact_science_capture_enabled
            and _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
                compact_snapshot_implementation
            )
        ):
            if not callable(
                getattr(runtime, "capture_native_snapshot_to_pinned_v1", None)
            ):
                raise CampaignProviderError(
                    "native snapshot implementation requires the pinned owner "
                    "capture seam"
                )
            try:
                if compact_snapshot_implementation == "native_capsule_v1":
                    encoder_extension = (
                        _native_snapshot_capsule.load_prebuilt_snapshot_extension_v1(
                            resolve_snapshot_extension_path_v1()
                        )
                    )
                else:
                    encoder_extension = provider._native_snapshot_encoder_loader()
                encoder_output = encoder_extension.manifest["output"]
                native_snapshot_encoder_identity = {
                    "artifact_path": str(encoder_extension.artifact_path),
                    "artifact_sha256": encoder_output["sha256"],
                    "manifest_path": str(encoder_extension.manifest_path),
                    "manifest_sha256": encoder_extension.manifest[
                        "manifest_sha256"
                    ],
                }
                _snapshot_seed, snapshot_world_uids = _seed_and_world_uids_v1(
                    request
                )
                publication_context_sha256 = (
                    _native_snapshot_execution_publication_context_sha256_v1(
                        plan=plan,
                        request=request,
                        context=context,
                        native_artifact=artifact,
                        encoder_extension=encoder_extension,
                        snapshot_implementation=compact_snapshot_implementation,
                    )
                )
                semantic_snapshot_plan = compile_cuda_snapshot_plan_v1(
                    world_uids=snapshot_world_uids,
                    world_cells=science_cells,
                    native_world_control_set_sha256=str(
                        native_world_control_set_sha256
                    ),
                    execution_publication_context_sha256=(
                        publication_context_sha256
                    ),
                    catalog_absolute_tick=(
                        contract.start_next_tick
                        if contract.start_next_tick in compact_snapshot_ticks
                        else None
                    ),
                )
                if compact_snapshot_implementation == "native_capsule_v1":
                    # Keep the Python semantic plan as the service contract.
                    # The native capsule compiler is passed to the publication
                    # service and binds that plan once at service creation;
                    # leaking the C++ plan object here would break the Python
                    # lifecycle/evidence authority and its mapping API.
                    compiled_snapshot_plan = semantic_snapshot_plan
                    native_plan_compiler = (
                        _native_snapshot_capsule.compile_snapshot_plan_v1
                    )
                    owned_encoder = getattr(
                        _native_snapshot_capsule,
                        "write_owned_compact_snapshot_to_v1_gzip_stage_file_v1",
                    )
                else:
                    compiled_snapshot_plan = semantic_snapshot_plan
                    native_plan_compiler = None
                    owned_encoder = None
                compiled_boundary_ticks = tuple(
                    int(row["absolute_tick"])
                    for row in compiled_snapshot_plan.to_mapping()[
                        "snapshot_boundaries"
                    ]
                )
                if compiled_boundary_ticks != compact_snapshot_ticks:
                    raise CampaignProviderError(
                        "native snapshot plan boundary union differs from the "
                        "provider schedule"
                    )
                factory_options: dict[str, Any] = {}
                if native_plan_compiler is not None:
                    factory_options.update(
                        {
                            "native_plan_compiler": native_plan_compiler,
                            "owned_encoder": owned_encoder,
                            "implementation_label": compact_snapshot_implementation,
                        }
                    )
                native_snapshot_pipeline = (
                    provider._native_snapshot_publication_factory(
                        compiled_plan=compiled_snapshot_plan,
                        runtime_owner=runtime,
                        slot_count=provider._native_snapshot_slots,
                        loaded_extension=encoder_extension,
                        **factory_options,
                    )
                )
            except CampaignProviderError:
                raise
            except Exception as exc:
                raise CampaignProviderError(
                    "native snapshot plan/service binding failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        if request.acs_checkpoint_slots:
            if request.acs_checkpoint_slots not in {1, 2}:
                raise CampaignProviderError(
                    "CUDA Native V2 detached ACS supports exactly one or two slots"
                )
            if request.native_streamed_checkpoint is not True:
                raise CampaignProviderError(
                    "CUDA Native V2 detached ACS requires native checkpointing"
                )
            from .acs_native_detached_slots_v1 import ACSNativeDetachedSlotsV1

            acs_pipeline = ACSNativeDetachedSlotsV1(
                slots=request.acs_checkpoint_slots
            )
        if compact_science_capture_enabled:
            compact_publication_pipeline = CompactScienceAsyncPublicationV1(
                workers=provider._compact_science_async_workers,
                pending_slots=provider._compact_science_async_pending_slots,
            )
    except BaseException:
        if native_snapshot_pipeline is not None:
            native_snapshot_pipeline.abort()
            native_snapshot_pipeline = None
        if compact_publication_pipeline is not None:
            compact_publication_pipeline.abort()
            compact_publication_pipeline = None
        if runtime is not None:
            runtime.close()
            runtime = None
        raise
    target = contract.start_next_tick + contract.total_ticks
    checkpoint_ticks = tuple(request.checkpoint_ticks)
    # A fresh owner may resume exactly at the terminal checkpoint.  In that
    # zero-dispatch case the restored checkpoint remains the authoritative
    # terminal result; requiring a newly emitted checkpoint would turn a safe
    # interruption between physical batches into an unrecoverable campaign.
    latest_checkpoint: dict[str, Any] | None = (
        dict(resume_checkpoint) if resume_checkpoint is not None else None
    )
    event_receipts: list[dict[str, Any]] = []

    def append_science_checkpoint_marker_v1(
        *,
        checkpoint_occurrence_id: str,
        absolute_tick: int,
        state_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        """Append one checkpoint marker and publish the current sidecar frontier."""

        if not science_capture_enabled:
            return None
        from engine.native_runtime.cuda_v2_science_capture_v1 import (
            build_native_cuda_v2_checkpoint_marker_output_v1,
        )

        for cell_id in sorted(science_records_by_cell):
            rows = science_records_by_cell[cell_id]
            control_binding_digest = _science_control_binding_digest_v1(
                rows,
                expected_native_control=native_control,
            )
            prefix = canonical_science_record_stream_sha256_v1(tuple(rows)).to_dict()
            state_digest = science_state_sha_by_cell.get(
                cell_id,
                {
                    "algorithm": "sha256",
                    "hex": str(
                        state_sha256
                        if state_sha256 is not None
                        else latest_checkpoint["state_sha256"]
                    ),
                },
            )
            marker_output = build_native_cuda_v2_checkpoint_marker_output_v1(
                cell_id=cell_id,
                absolute_tick=absolute_tick,
                checkpoint_occurrence_id=checkpoint_occurrence_id,
                authoritative_state_sha256=state_digest,
                observation_prefix_sha256=prefix,
                next_world_local_sequence=len(rows) + 1,
                continuation_parent_identity=parent_occurrence_digest,
                control_binding_sha256=control_binding_digest,
            )
            marker_records = (
                _science_observation_projection.finalize_native_cuda_v2_observations_v1(
                    marker_output,
                    context=_science_context_v1(science_cells[cell_id]),
                    start_world_local_sequence=len(rows),
                )
            )
            science_records_by_cell[cell_id].extend(marker_records)
        science_checkpoint_occurrences.append(checkpoint_occurrence_id)
        return _science_sidecar_snapshot_v1(
            context=context,
            request=request,
            records_by_cell=science_records_by_cell,
            next_absolute_tick=absolute_tick,
            terminal=False,
        )

    def persist_compact_snapshot_v1(*, absolute_tick: int) -> None:
        nonlocal native_snapshot_recovered_existing_count
        if not compact_science_capture_enabled:
            return
        phase_started = perf_counter()
        boundary_started_ns = (
            perf_counter_ns() if compact_snapshot_timing_enabled else 0
        )
        boundary_cpu_started_ns = (
            process_time_ns() if compact_snapshot_timing_enabled else 0
        )
        sample: dict[str, Any] | None = (
            {
                "schema_version": "eiel_compact_science_snapshot_boundary_timing_v1",
                "claim_class": "TECHNICAL_ONLY",
                "implementation": compact_snapshot_implementation,
                "absolute_tick": absolute_tick,
                "phases": {},
            }
            if compact_snapshot_timing_enabled
            else None
        )
        memory_before: dict[str, Any] = {}
        if sample is not None:
            memory_before["process_rss_high_water_bytes"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * 1024
            try:
                import torch

                if torch.cuda.is_available():
                    memory_before.update(
                        {
                            "gpu_allocator_allocated_current_bytes": int(
                                torch.cuda.memory_allocated()
                            ),
                            "gpu_allocator_reserved_current_bytes": int(
                                torch.cuda.memory_reserved()
                            ),
                        }
                    )
            except (ImportError, RuntimeError):
                memory_before["gpu_allocator_current_bytes"] = None
        settlement_started_ns = (
            perf_counter_ns() if compact_snapshot_timing_enabled else 0
        )
        settlement_cpu_started_ns = (
            process_time_ns() if compact_snapshot_timing_enabled else 0
        )
        settle_all_compact_publications_v1(producer_backpressure=True)
        if sample is not None:
            sample["phases"]["prior_dispatch_fifo_settlement"] = {
                "wall_time_ns": perf_counter_ns() - settlement_started_ns,
                "cpu_time_ns": process_time_ns() - settlement_cpu_started_ns,
            }
        destination = (
            Path(context.batch_root)
            / "science-compact"
            / "snapshots"
            / f"tick-{absolute_tick:012d}"
        )
        if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
            compact_snapshot_implementation
        ):
            if native_snapshot_pipeline is None:
                raise CampaignProviderError(
                    "native snapshot publication service is absent"
                )
            if compiled_snapshot_plan is None:
                raise CampaignProviderError(
                    "native snapshot publication plan is absent"
                )
            if os.path.lexists(destination):
                try:
                    existing = (
                        _native_snapshot_publication
                        .inspect_existing_native_snapshot_root_v1(
                            destination,
                            compiled_plan=compiled_snapshot_plan,
                            absolute_tick=absolute_tick,
                        )
                    )
                    inventory_state = (
                        _reconcile_compact_science_inventory_commit_v1(
                            context,
                            destination,
                            artifact_family=(
                                "cuda_full_evolution_v2.compact_science_snapshot"
                            ),
                        )
                    )
                    if inventory_state == "COMMITTED_EXACT_DURABLE":
                        pointer = _compact_science_artifact_pointer_v1(existing)
                        if pointer not in compact_snapshot_receipts:
                            compact_snapshot_receipts.append(pointer)
                        native_snapshot_recovered_existing_count += 1
                        compact_phase_timing["compact_snapshot_seconds"] += (
                            perf_counter() - phase_started
                        )
                        return
                    if inventory_state != "NOT_COMMITTED_REMOVE_ROOT":
                        raise CampaignProviderError(
                            "existing native snapshot publication has lifecycle "
                            f"state {inventory_state}"
                        )
                    _native_snapshot_publication.remove_native_snapshot_root_v1(
                        destination
                    )
                    if os.path.lexists(destination):
                        raise CampaignProviderError(
                            "known-uncommitted native snapshot root cleanup did "
                            "not converge"
                        )
                except CampaignProviderError:
                    raise
                except Exception as exc:
                    raise CampaignProviderError(
                        "existing native snapshot publication reconciliation "
                        f"failed: {type(exc).__name__}: {exc}"
                    ) from exc
            submit_started_ns = (
                perf_counter_ns() if compact_snapshot_timing_enabled else 0
            )
            submit_cpu_started_ns = (
                process_time_ns() if compact_snapshot_timing_enabled else 0
            )
            try:
                settle_ready_native_snapshots_v1()
                while not native_snapshot_pipeline.has_capacity:
                    settle_oldest_native_snapshot_v1(
                        producer_backpressure=True
                    )
                token = native_snapshot_pipeline.submit(
                    runtime,
                    absolute_tick,
                    destination,
                )
            except Exception as exc:
                if sample is not None:
                    sample["success"] = False
                    sample["failure"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    producer_elapsed_ns = perf_counter_ns() - boundary_started_ns
                    sample["producer_critical_path_time_ns"] = producer_elapsed_ns
                    sample["full_durable_completion_time_ns"] = (
                        producer_elapsed_ns
                    )
                    sample["publication_lag_time_ns"] = 0
                    sample["gpu_sync_host_wait_upper_bound_ns"] = (
                        producer_elapsed_ns
                    )
                    sample["evolution_producer_blocking_time_ns"] = (
                        producer_elapsed_ns
                    )
                    compact_snapshot_timing_samples.append(sample)
                raise CampaignProviderError(
                    "native compact science snapshot admission failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            handoff_completed_ns = perf_counter_ns()
            if sample is not None:
                sample["phases"]["native_capture_and_submit"] = {
                    "wall_time_ns": handoff_completed_ns - submit_started_ns,
                    "cpu_time_ns": process_time_ns() - submit_cpu_started_ns,
                }
                producer_elapsed_ns = handoff_completed_ns - boundary_started_ns
                sample["producer_critical_path_time_ns"] = producer_elapsed_ns
                # Any synchronous GPU work hidden inside owner validation is
                # bounded above by the whole producer handoff until the owner
                # exposes a narrower per-submission wait metric.
                sample["gpu_sync_host_wait_upper_bound_ns"] = producer_elapsed_ns
                sample["evolution_producer_blocking_time_ns"] = (
                    producer_elapsed_ns
                )
            pending_native_snapshots.append(
                {
                    "token": token,
                    "sample": sample,
                    "boundary_started_ns": boundary_started_ns,
                    "handoff_completed_ns": handoff_completed_ns,
                }
            )
            compact_phase_timing["compact_snapshot_seconds"] += (
                perf_counter() - phase_started
            )
            return
        state_timing: dict[str, Any] = {}
        metrics_timing: dict[str, Any] = {}
        writer_timing: dict[str, Any] = {}
        try:
            projection_started_ns = (
                perf_counter_ns() if compact_snapshot_timing_enabled else 0
            )
            projection_cpu_started_ns = (
                process_time_ns() if compact_snapshot_timing_enabled else 0
            )
            state_projection = (
                runtime.state_projection(timing_diagnostics=state_timing)
                if compact_snapshot_timing_enabled
                else runtime.state_projection()
            )
            if sample is not None:
                sample["phases"]["state_projection_provider_boundary"] = {
                    "wall_time_ns": perf_counter_ns() - projection_started_ns,
                    "cpu_time_ns": process_time_ns()
                    - projection_cpu_started_ns,
                }
            metrics_started_ns = (
                perf_counter_ns() if compact_snapshot_timing_enabled else 0
            )
            metrics_cpu_started_ns = (
                process_time_ns() if compact_snapshot_timing_enabled else 0
            )
            active_metrics_projection = (
                runtime.compact_science_active_metrics_projection_v1(
                    timing_diagnostics=metrics_timing
                )
                if compact_snapshot_timing_enabled
                else runtime.compact_science_active_metrics_projection_v1()
            )
            if sample is not None:
                sample["phases"]["metrics_projection_provider_boundary"] = {
                    "wall_time_ns": perf_counter_ns() - metrics_started_ns,
                    "cpu_time_ns": process_time_ns() - metrics_cpu_started_ns,
                }
            writer_started_ns = (
                perf_counter_ns() if compact_snapshot_timing_enabled else 0
            )
            writer_cpu_started_ns = (
                process_time_ns() if compact_snapshot_timing_enabled else 0
            )
            receipt = _compact_science_capture.write_compact_science_snapshot_v1(
                destination,
                state=state_projection,
                active_metrics_projection=active_metrics_projection,
                world_cells=science_cells,
                absolute_tick=absolute_tick,
                native_world_control_set_sha256=str(
                    native_world_control_set_sha256
                ),
                include_genome_catalog=(
                    absolute_tick == contract.start_next_tick
                ),
                timing_diagnostics=(
                    writer_timing if compact_snapshot_timing_enabled else None
                ),
            )
            if sample is not None:
                sample["phases"]["snapshot_writer_provider_boundary"] = {
                    "wall_time_ns": perf_counter_ns() - writer_started_ns,
                    "cpu_time_ns": process_time_ns() - writer_cpu_started_ns,
                }
        except Exception as exc:
            if sample is not None:
                sample["success"] = False
                sample["failure"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                sample["producer_critical_path_time_ns"] = (
                    perf_counter_ns() - boundary_started_ns
                )
                sample["full_durable_completion_time_ns"] = sample[
                    "producer_critical_path_time_ns"
                ]
                sample["publication_lag_time_ns"] = 0
                compact_snapshot_timing_samples.append(sample)
            raise CampaignProviderError(
                f"compact science snapshot materialization failed: {exc}"
            ) from exc
        lifecycle_started_ns = (
            perf_counter_ns() if compact_snapshot_timing_enabled else 0
        )
        lifecycle_cpu_started_ns = (
            process_time_ns() if compact_snapshot_timing_enabled else 0
        )
        _record_compact_science_root_v1(
            context,
            receipt["path"],
            artifact_family="cuda_full_evolution_v2.compact_science_snapshot",
        )
        if sample is not None:
            sample["phases"]["manifest_evidence_lifecycle_registration"] = {
                "wall_time_ns": perf_counter_ns() - lifecycle_started_ns,
                "cpu_time_ns": process_time_ns() - lifecycle_cpu_started_ns,
            }
        pointer_started_ns = (
            perf_counter_ns() if compact_snapshot_timing_enabled else 0
        )
        pointer_cpu_started_ns = (
            process_time_ns() if compact_snapshot_timing_enabled else 0
        )
        compact_snapshot_receipts.append(
            _compact_science_artifact_pointer_v1(receipt)
        )
        if sample is not None:
            sample["phases"]["receipt_pointer_append"] = {
                "wall_time_ns": perf_counter_ns() - pointer_started_ns,
                "cpu_time_ns": process_time_ns() - pointer_cpu_started_ns,
            }
            for prefix, component in (
                ("state_projection", state_timing),
                ("metrics_projection", metrics_timing),
                ("writer", writer_timing),
            ):
                component_phases = component.get("phases", {})
                if isinstance(component_phases, Mapping):
                    for name, phase in component_phases.items():
                        if isinstance(phase, Mapping):
                            sample["phases"][f"{prefix}.{name}"] = dict(phase)
            writer_counters = writer_timing.get("counters", {})
            state_d2h_ns = int(
                writer_timing.get("phases", {})
                .get("state_d2h_transfer_and_wait", {})
                .get("wall_time_ns", 0)
            )
            metrics_d2h_ns = int(
                metrics_timing.get("phases", {})
                .get("metrics_d2h_and_contiguous", {})
                .get("wall_time_ns", 0)
            )
            total_wall_ns = perf_counter_ns() - boundary_started_ns
            sample.update(
                {
                    "success": True,
                    "artifact_sha256": receipt["artifact_sha256"],
                    "path": receipt["path"],
                    "producer_critical_path_time_ns": total_wall_ns,
                    "full_durable_completion_time_ns": total_wall_ns,
                    "publication_lag_time_ns": 0,
                    "gpu_sync_host_wait_upper_bound_ns": (
                        state_d2h_ns + metrics_d2h_ns
                    ),
                    "evolution_producer_blocking_time_ns": total_wall_ns,
                    "boundary_cpu_time_ns": process_time_ns()
                    - boundary_cpu_started_ns,
                    "counters": {
                        "row_count": int(writer_counters.get("row_count", 0)),
                        "captured_world_count": int(
                            writer_counters.get("captured_world_count", 0)
                        ),
                        "active_agent_count": int(
                            writer_counters.get("active_agent_count", 0)
                        ),
                        "d2h_bytes": int(
                            writer_counters.get("state_host_bytes", 0)
                        )
                        + int(metrics_timing.get("d2h_bytes", 0)),
                        "uncompressed_bytes": int(
                            writer_counters.get("uncompressed_bytes", 0)
                        ),
                        "compressed_bytes": int(
                            writer_counters.get("compressed_bytes", 0)
                        ),
                        "records_per_second_durable": (
                            int(writer_counters.get("row_count", 0))
                            * 1_000_000_000
                            / total_wall_ns
                        ),
                        "uncompressed_megabytes_per_second_durable": (
                            int(writer_counters.get("uncompressed_bytes", 0))
                            * 1_000_000_000
                            / (1024 * 1024)
                            / total_wall_ns
                        ),
                        "snapshot_queue_depth": 0,
                        "snapshot_queue_depth_high_water": 0,
                        "buffer_reuse_count": 0,
                    },
                    "memory": {
                        "before": memory_before,
                        "process_rss_high_water_after_bytes": int(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        )
                        * 1024,
                        "pinned_host_bytes": 0,
                        "native_heap_bytes": None,
                        "snapshot_resident_buffer_count": 0,
                    },
                }
            )
            try:
                import torch

                if torch.cuda.is_available():
                    sample["memory"].update(
                        {
                            "gpu_allocator_allocated_high_water_bytes": int(
                                torch.cuda.max_memory_allocated()
                            ),
                            "gpu_allocator_reserved_high_water_bytes": int(
                                torch.cuda.max_memory_reserved()
                            ),
                            "gpu_allocator_allocated_after_bytes": int(
                                torch.cuda.memory_allocated()
                            ),
                            "gpu_allocator_reserved_after_bytes": int(
                                torch.cuda.memory_reserved()
                            ),
                        }
                    )
            except (ImportError, RuntimeError):
                sample["memory"]["gpu_allocator_high_water_bytes"] = None
            compact_snapshot_timing_samples.append(sample)
        compact_phase_timing["compact_snapshot_seconds"] += (
            perf_counter() - phase_started
        )

    def persist_compact_checkpoint_frontier_v1(
        *, absolute_tick: int
    ) -> Mapping[str, Any] | None:
        nonlocal compact_checkpoint_frontier
        if not compact_science_capture_enabled:
            return None
        phase_started = perf_counter()
        settle_all_compact_publications_v1(producer_backpressure=True)
        # A continuation frontier may reference only snapshots whose native
        # bytes, legacy manifest, lifecycle inventory, and pointer are already
        # durable.  Drain before sampling the counter frontier so checkpoint
        # authority cannot overtake snapshot publication.
        settle_all_native_snapshots_v1(producer_backpressure=True)
        destination = (
            Path(context.batch_root)
            / "science-compact"
            / "checkpoint-frontiers"
            / f"tick-{absolute_tick:012d}"
        )
        try:
            receipt = (
                _compact_science_capture.write_compact_science_checkpoint_frontier_v1(
                destination,
                active_metrics_projection=(
                    runtime.compact_science_active_metrics_projection_v1()
                ),
                absolute_tick=absolute_tick,
                native_world_control_set_sha256=str(
                    native_world_control_set_sha256
                ),
                )
            )
        except Exception as exc:
            raise CampaignProviderError(
                f"compact science checkpoint frontier failed: {exc}"
            ) from exc
        _record_compact_science_root_v1(
            context,
            receipt["path"],
            artifact_family=(
                "cuda_full_evolution_v2.compact_science_checkpoint_frontier"
            ),
        )
        compact_checkpoint_frontier = dict(receipt)
        compact_phase_timing["compact_checkpoint_frontier_seconds"] += (
            perf_counter() - phase_started
        )
        return _compact_science_continuation_frontier_v1(
            counter_frontier=receipt,
            dispatch_receipts=compact_dispatch_receipts,
            snapshot_receipts=compact_snapshot_receipts,
            native_world_control_set_sha256=str(
                native_world_control_set_sha256
            ),
        )

    def record_direct_sparse_artifacts_v1(event_receipt: Mapping[str, Any]) -> None:
        """Publish lifecycle inventory only from the provider owner thread."""

        if event_receipt.get("durable_envelope_released") is not True:
            context.record_artifact(
                event_receipt["durable_envelope_path"],
                artifact_family=(
                    "cuda_full_evolution_v2.direct_sparse_event_envelope"
                ),
                retention_class="R0",
                release_predicate=(
                    "terminal checkpoint and durable sink acknowledgement sealed"
                ),
                replacement_verified=False,
            )
        context.record_artifact(
            Path(str(event_receipt["path"])) / "sink_receipt.json",
            artifact_family=(
                "cuda_full_evolution_v2.direct_sparse_sink_receipt"
            ),
            retention_class="R5",
            release_predicate="durable event acknowledgement authority",
            replacement_verified=False,
        )

    def settle_oldest_compact_publication_v1(
        *, producer_backpressure: bool
    ) -> None:
        """Promote, inventory, and acknowledge one exact FIFO host stage."""

        if (
            compact_publication_pipeline is None
            or not pending_compact_event_receipts
        ):
            raise CampaignProviderError(
                "compact science async settlement has no FIFO head"
        )
        token = compact_publication_pipeline.head_token
        event_receipt = pending_compact_event_receipts[0]
        phase_started = perf_counter()
        try:
            _, compact_pointer = (
                _settle_compact_science_publication_transaction_v1(
                    pipeline=compact_publication_pipeline,
                    token=token,
                    event_receipt=event_receipt,
                    direct_sparse=direct_sparse,
                    context=context,
                    producer_backpressure=producer_backpressure,
                )
            )
            compact_dispatch_receipts.append(compact_pointer)
            pending_compact_event_receipts.popleft()
        except CampaignProviderError:
            raise
        except Exception as exc:
            raise CampaignProviderError(
                "compact science async publication settlement failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            compact_phase_timing["compact_fifo_settlement_seconds"] += (
                perf_counter() - phase_started
            )

    def settle_ready_compact_publications_v1() -> None:
        if compact_publication_pipeline is None:
            return
        compact_publication_pipeline.poll_failure()
        while pending_compact_event_receipts:
            token = compact_publication_pipeline.head_token
            if not compact_publication_pipeline.ready(token):
                break
            settle_oldest_compact_publication_v1(
                producer_backpressure=False
            )

    def settle_all_compact_publications_v1(
        *, producer_backpressure: bool
    ) -> None:
        while pending_compact_event_receipts:
            settle_oldest_compact_publication_v1(
                producer_backpressure=producer_backpressure
            )

    def close_compact_publication_v1() -> None:
        nonlocal compact_publication_telemetry
        if compact_publication_pipeline is None:
            return
        settle_all_compact_publications_v1(producer_backpressure=True)
        compact_publication_pipeline.close()
        compact_publication_telemetry = compact_publication_pipeline.telemetry()
        compact_publication_telemetry["provider_phase_timing"] = {
            **compact_phase_timing,
            "compact_execution_elapsed_seconds": (
                perf_counter() - compact_phase_started
            ),
        }

    def settle_oldest_native_snapshot_v1(
        *, producer_backpressure: bool
    ) -> None:
        """Publish, inventory, and acknowledge one native snapshot FIFO head."""

        if native_snapshot_pipeline is None or not pending_native_snapshots:
            raise CampaignProviderError(
                "native snapshot settlement has no FIFO head"
            )
        pending = pending_native_snapshots[0]
        token = pending["token"]
        if native_snapshot_pipeline.head_token != token:
            raise CampaignProviderError(
                "native snapshot provider/service FIFO identity differs"
            )
        sample = pending.get("sample")
        settlement_started_ns = perf_counter_ns()
        settlement_cpu_started_ns = process_time_ns()
        try:
            receipt, pointer, timing = (
                _settle_native_snapshot_publication_transaction_v1(
                    pipeline=native_snapshot_pipeline,
                    token=token,
                    context=context,
                    producer_backpressure=producer_backpressure,
                )
            )
            pointer_started_ns = perf_counter_ns()
            compact_snapshot_receipts.append(pointer)
            pointer_elapsed_ns = perf_counter_ns() - pointer_started_ns
            pending_native_snapshots.popleft()
        except Exception as exc:
            if isinstance(sample, dict):
                completed_ns = perf_counter_ns()
                settlement_elapsed_ns = completed_ns - settlement_started_ns
                sample["success"] = False
                sample["failure"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                sample["full_durable_completion_time_ns"] = (
                    completed_ns - int(pending["boundary_started_ns"])
                )
                sample["publication_lag_time_ns"] = (
                    completed_ns - int(pending["handoff_completed_ns"])
                )
                sample["evolution_producer_blocking_time_ns"] = int(
                    sample["evolution_producer_blocking_time_ns"]
                ) + settlement_elapsed_ns
                sample["phases"]["native_fifo_settlement_failure"] = {
                    "wall_time_ns": settlement_elapsed_ns,
                    "cpu_time_ns": (
                        process_time_ns() - settlement_cpu_started_ns
                    ),
                }
                compact_snapshot_timing_samples.append(sample)
            if isinstance(exc, CampaignProviderError):
                raise
            raise CampaignProviderError(
                "native snapshot FIFO settlement failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(sample, dict):
            return

        completed_ns = perf_counter_ns()
        settlement_elapsed_ns = completed_ns - settlement_started_ns
        phases = sample["phases"]
        for name in (
            "native_fifo_result_wait_ns",
            "atomic_publication_ns",
            "manifest_evidence_lifecycle_registration_ns",
            "publication_acknowledgement_ns",
        ):
            value = timing.get(name)
            if type(value) is int:
                phases[name.removesuffix("_ns")] = {"wall_time_ns": value}
        phases["receipt_pointer_append"] = {
            "wall_time_ns": pointer_elapsed_ns,
        }
        capture_metrics = timing.get("capture_metrics")
        producer_handoff_ns = int(sample["producer_critical_path_time_ns"])
        sample["producer_handoff_time_ns"] = producer_handoff_ns
        if isinstance(capture_metrics, Mapping):
            capture_event_ns = capture_metrics.get("cuda_event_time_ns")
            if type(capture_event_ns) is int and capture_event_ns >= 0:
                phases["native_d2h_copy_event"] = {
                    "wall_time_ns": capture_event_ns,
                    "cuda_event_time_ns": capture_event_ns,
                }
                serialized_critical_path_ns = (
                    producer_handoff_ns + capture_event_ns
                )
                sample["producer_critical_path_time_ns"] = (
                    serialized_critical_path_ns
                )
                sample["gpu_sync_host_wait_upper_bound_ns"] = (
                    serialized_critical_path_ns
                )
                sample["evolution_producer_blocking_time_ns"] = int(
                    sample["evolution_producer_blocking_time_ns"]
                ) + capture_event_ns
        for prefix, values in (
            ("native_encoder", timing.get("encoder_timing")),
            ("native_worker_publication", timing.get("worker_publication_timing")),
        ):
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if type(value) is int and str(name).endswith("_ns"):
                    phases[f"{prefix}.{name.removesuffix('_ns')}"] = {
                        "wall_time_ns": value,
                    }
        counters = dict(timing.get("native_counters", {}))
        counters.update(
            {
                "row_count": receipt["row_count"],
                "captured_world_count": receipt["captured_world_count"],
                "compressed_bytes": receipt["compressed_bytes"],
            }
        )
        sample["counters"] = counters
        sample["success"] = True
        sample["full_durable_completion_time_ns"] = (
            completed_ns - int(pending["boundary_started_ns"])
        )
        sample["publication_lag_time_ns"] = (
            completed_ns - int(pending["handoff_completed_ns"])
        )
        sample["evolution_producer_blocking_time_ns"] = int(
            sample["evolution_producer_blocking_time_ns"]
        ) + settlement_elapsed_ns
        sample["memory"] = dict(native_snapshot_pipeline.telemetry())
        compact_snapshot_timing_samples.append(sample)

    def settle_ready_native_snapshots_v1() -> None:
        if native_snapshot_pipeline is None:
            return
        native_snapshot_pipeline.poll_failure()
        while pending_native_snapshots:
            token = pending_native_snapshots[0]["token"]
            if not native_snapshot_pipeline.ready(token):
                break
            settle_oldest_native_snapshot_v1(producer_backpressure=False)

    def settle_all_native_snapshots_v1(
        *, producer_backpressure: bool
    ) -> None:
        while pending_native_snapshots:
            settle_oldest_native_snapshot_v1(
                producer_backpressure=producer_backpressure
            )

    def close_native_snapshot_publication_v1() -> None:
        nonlocal native_snapshot_publication_telemetry
        if native_snapshot_pipeline is None:
            return
        settle_all_native_snapshots_v1(producer_backpressure=True)
        native_snapshot_pipeline.close()
        native_snapshot_publication_telemetry = dict(
            native_snapshot_pipeline.telemetry()
        )
        native_snapshot_publication_telemetry["encoder_artifact"] = (
            native_snapshot_encoder_identity
        )
        native_snapshot_publication_telemetry[
            "recovered_existing_publication_count"
        ] = native_snapshot_recovered_existing_count
        native_snapshot_publication_telemetry["provider_phase_timing"] = {
            **compact_phase_timing,
            "compact_execution_elapsed_seconds": (
                perf_counter() - compact_phase_started
            ),
        }

    def settle_oldest_acs_checkpoint(*, producer_backpressure: bool) -> None:
        """Publish one durable FIFO head through CRCA and lifecycle authority."""

        nonlocal latest_checkpoint, parent_occurrence_digest
        if acs_pipeline is None or not pending_acs_checkpoints:
            raise CampaignProviderError("CUDA Native V2 ACS settlement has no FIFO head")
        pending = pending_acs_checkpoints[0]
        token = pending["token"]
        detached = acs_pipeline.result(
            token,
            producer_backpressure=producer_backpressure,
        )
        expected_compact_frontier = pending.get("compact_science_frontier")
        detached_compact_frontier = detached.get("compact_science_frontier")
        if compact_science_capture_enabled:
            if (
                not isinstance(expected_compact_frontier, Mapping)
                or detached_compact_frontier != expected_compact_frontier
            ):
                raise CampaignProviderError(
                    "ACS2 checkpoint did not return the exact bound compact "
                    "science frontier"
                )
        elif detached_compact_frontier is not None:
            raise CampaignProviderError(
                "ACS2 checkpoint unexpectedly carries a compact science frontier"
            )
        checkpoint_tick = int(pending["checkpoint_at_tick"])
        bundle = Path(str(detached["path"])).resolve()
        checkpoint_id = str(detached["checkpoint_id"])
        state_sha256 = str(detached["canonical_state_sha256"])
        occurrence_root: Path | None = None
        try:
            tree_digest = _seal_published_checkpoint_bundle_v1(
                bundle,
                checkpoint_id=checkpoint_id,
                absolute_next_tick=checkpoint_tick,
                context=context,
                occurrence_adapter=occurrence_adapter,
            )
            occurrence_digest, occurrence_root = (
                _materialize_authoritative_occurrence_v1(
                    bundle,
                    checkpoint_id=checkpoint_id,
                    tree_digest=tree_digest,
                    absolute_next_tick=checkpoint_tick,
                    context=context,
                    source_batch_id=request.batch_id,
                    attempt_authority=attempt_authority,
                    parent_checkpoint_digest=parent_occurrence_digest,
                    occurrence_adapter=occurrence_adapter,
                )
            )
            science_observation_frontier = append_science_checkpoint_marker_v1(
                checkpoint_occurrence_id=(
                    str(occurrence_digest["hex"])
                    if occurrence_digest is not None
                    else checkpoint_id
                ),
                absolute_tick=checkpoint_tick,
                state_sha256=state_sha256,
            )
            latest_checkpoint = _checkpoint_record_v1(
                request=request,
                context=context,
                bundle=bundle,
                checkpoint_id=checkpoint_id,
                tree_digest=tree_digest,
                state_sha256=state_sha256,
                absolute_next_tick=checkpoint_tick,
                occurrence_digest=occurrence_digest,
                occurrence_root=occurrence_root,
                completed_restore_ticks=completed_restore_ticks,
                science_continuation_admission=_base_checkpoint_science_admission_v1(
                    artifact
                ),
                science_observation_frontier=science_observation_frontier,
                compact_science_frontier=(
                    None
                    if detached_compact_frontier is None
                    else dict(detached_compact_frontier)
                ),
                native_control=native_control,
            )
            context.record_checkpoint(latest_checkpoint)
        except BaseException as exc:
            _quarantine_after_external_commit_failure_v1(
                occurrence_root,
                exc,
                label="checkpoint occurrence",
            )
            raise
        if occurrence_digest is not None:
            parent_occurrence_digest = occurrence_digest
        acs_checkpoint_receipts.append(dict(detached))
        acs_pipeline.acknowledge_durable(token)
        pending_acs_checkpoints.popleft()

    def settle_ready_acs_checkpoints() -> None:
        if acs_pipeline is None:
            return
        while pending_acs_checkpoints and acs_pipeline.ready(
            pending_acs_checkpoints[0]["token"]
        ):
            settle_oldest_acs_checkpoint(producer_backpressure=False)

    def settle_all_acs_checkpoints(*, producer_backpressure: bool) -> None:
        while pending_acs_checkpoints:
            settle_oldest_acs_checkpoint(
                producer_backpressure=producer_backpressure
            )

    def interrupt_for_progressive_storage_v1(
        blocked: StorageBudgetExceeded,
        *,
        absolute_tick: int,
    ) -> dict[str, Any]:
        """Publish a resumable owner-transition checkpoint before releasing GPU.

        The preceding successful incremental reservation includes this exact
        checkpoint/frontier bound.  A failed reservation for the *next*
        dispatch therefore cannot consume the emergency continuation space.
        """

        nonlocal latest_checkpoint, parent_occurrence_digest, runtime
        nonlocal direct_sparse_execution_receipt
        if (
            progressive_storage_guard is None
            or runtime is None
            or acs_pipeline is not None
            or not direct_sparse
            or not compact_science_capture_enabled
        ):
            raise CampaignProviderError(
                "progressive storage interruption is unavailable for this route"
            ) from blocked
        compact_science_continuation = persist_compact_checkpoint_frontier_v1(
            absolute_tick=absolute_tick
        )
        sealed = runtime.checkpoint()
        bundle, checkpoint_id, tree_digest, state_sha256 = (
            _publish_checkpoint_bundle_v1(
                sealed,
                context=context,
                artifact=artifact,
                occurrence_adapter=occurrence_adapter,
            )
        )
        occurrence_root: Path | None = None
        try:
            occurrence_digest, occurrence_root = (
                _materialize_authoritative_occurrence_v1(
                    bundle,
                    checkpoint_id=checkpoint_id,
                    tree_digest=tree_digest,
                    absolute_next_tick=absolute_tick,
                    context=context,
                    source_batch_id=request.batch_id,
                    attempt_authority=attempt_authority,
                    parent_checkpoint_digest=parent_occurrence_digest,
                    occurrence_adapter=occurrence_adapter,
                )
            )
            science_observation_frontier = append_science_checkpoint_marker_v1(
                checkpoint_occurrence_id=(
                    str(occurrence_digest["hex"])
                    if occurrence_digest is not None
                    else checkpoint_id
                ),
                absolute_tick=absolute_tick,
                state_sha256=state_sha256,
            )
            latest_checkpoint = _checkpoint_record_v1(
                request=request,
                context=context,
                bundle=bundle,
                checkpoint_id=checkpoint_id,
                tree_digest=tree_digest,
                state_sha256=state_sha256,
                absolute_next_tick=absolute_tick,
                occurrence_digest=occurrence_digest,
                occurrence_root=occurrence_root,
                completed_restore_ticks=completed_restore_ticks,
                science_continuation_admission=(
                    _base_checkpoint_science_admission_v1(artifact)
                ),
                science_observation_frontier=science_observation_frontier,
                compact_science_frontier=compact_science_continuation,
                native_control=native_control,
            )
            context.record_checkpoint(latest_checkpoint)
        except BaseException as exc:
            _quarantine_after_external_commit_failure_v1(
                occurrence_root,
                exc,
                label="progressive-storage checkpoint occurrence",
            )
            _quarantine_after_external_commit_failure_v1(
                bundle,
                exc,
                label="progressive-storage checkpoint bundle",
            )
            raise
        if occurrence_digest is not None:
            parent_occurrence_digest = occurrence_digest
        direct_sparse_execution_receipt = dict(runtime.execution_receipt_v3())
        close_native_snapshot_publication_v1()
        runtime.close()
        runtime = None
        close_compact_publication_v1()
        return {
            "status": "interrupted",
            "checkpoint": latest_checkpoint,
            "metrics_rows": [],
            "result": {
                "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                "next_absolute_tick": absolute_tick,
                "crca_final_frontiers": {
                    "last_dispatched_next_tick": absolute_tick,
                    "last_computed_next_tick": absolute_tick,
                    "durable_checkpoint_next_tick": absolute_tick,
                    "published_artifact_next_tick": absolute_tick,
                },
                "event_dispatch_receipts": event_receipts,
                "direct_sparse_execution_receipt": (
                    direct_sparse_execution_receipt
                ),
                "native_control": native_control,
                "science_continuation_admission": (
                    _base_checkpoint_science_admission_v1(artifact)
                ),
                "compact_science_frontier": compact_science_continuation,
                "compact_science_async_publication": (
                    compact_publication_telemetry
                ),
                **(
                    {
                        "compact_science_native_snapshot_publication": (
                            native_snapshot_publication_telemetry
                        )
                    }
                    if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
                        compact_snapshot_implementation
                    )
                    else {}
                ),
                "storage_wait": {
                    "status": "WAIT_STORAGE",
                    "resume_safe_boundary": "absolute_tick_owner_transition",
                    "next_absolute_tick": absolute_tick,
                    "guard_receipt": dict(blocked.receipt),
                },
                "technical_only": True,
                "formal_qualification_authority": False,
            },
        }

    execution_error: BaseException | None = None
    try:
        if progressive_storage_guard is not None:
            try:
                progressive_storage_guard.enforce(
                    stage=f"provider_before_initial_boundary:{current}"
                )
            except StorageBudgetExceeded as exc:
                raise CampaignProviderError(
                    "progressive storage admission changed before the first "
                    "native dispatch"
                ) from exc
        if (
            compact_science_capture_enabled
            and resume_checkpoint is None
            and current in compact_snapshot_ticks
        ):
            persist_compact_snapshot_v1(absolute_tick=current)
        while current < target:
            if compact_publication_pipeline is not None:
                settle_ready_compact_publications_v1()
                while not compact_publication_pipeline.has_capacity:
                    settle_oldest_compact_publication_v1(
                        producer_backpressure=True
                    )
            if native_snapshot_pipeline is not None:
                settle_ready_native_snapshots_v1()
                while not native_snapshot_pipeline.has_capacity:
                    settle_oldest_native_snapshot_v1(
                        producer_backpressure=True
                    )
            if progressive_storage_guard is not None:
                try:
                    progressive_storage_guard.enforce(
                        stage=f"provider_before_dispatch:{current}"
                    )
                except StorageBudgetExceeded as exc:
                    return interrupt_for_progressive_storage_v1(
                        exc,
                        absolute_tick=current,
                    )
            next_checkpoint = next(
                (tick for tick in checkpoint_ticks if tick > current), target
            )
            dispatch_ticks = min(
                contract.temporal_k, next_checkpoint - current, target - current
            )
            next_control_boundary = getattr(
                runtime, "next_native_world_control_boundary_v1", None
            )
            if (
                type(next_control_boundary) is int
                and next_control_boundary > current
            ):
                dispatch_ticks = min(
                    dispatch_ticks, next_control_boundary - current
                )
            next_snapshot = next(
                (tick for tick in compact_snapshot_ticks if tick > current),
                None,
            )
            if next_snapshot is not None:
                dispatch_ticks = min(dispatch_ticks, next_snapshot - current)
            if dispatch_ticks <= 0:
                raise CampaignProviderError("CUDA Native V2 dispatch made no progress")
            state_before = None
            science_capture = None
            compact_science_capture = None
            if science_capture_enabled:
                state_before = runtime.state_projection()
            dispatch_phase_started = perf_counter()
            if direct_sparse:
                if compact_science_capture_enabled:
                    dispatch, compact_science_capture = (
                        runtime.advance_and_drain_compact_science_v1(
                            dispatch_ticks
                        )
                    )
                elif science_capture_enabled:
                    dispatch, science_capture = runtime.advance_and_drain_science_v1(
                        dispatch_ticks
                    )
                else:
                    dispatch = runtime.advance_and_drain_v3(dispatch_ticks)
            else:
                runtime.advance(dispatch_ticks)
                dispatch = runtime.drain_events(to_cpu=True)
            compact_phase_timing["native_dispatch_and_capture_seconds"] += (
                perf_counter() - dispatch_phase_started
            )
            if (
                int(dispatch.start_next_absolute_tick) != current
                or int(dispatch.end_next_absolute_tick) != current + dispatch_ticks
            ):
                raise CampaignProviderError("CUDA Native V2 dispatch frontier differs")
            durable_phase_started = perf_counter()
            if direct_sparse:
                event_receipt = _persist_direct_sparse_event_dispatch_v3(
                    runtime,
                    dispatch,
                    scratch_root=Path(context.batch_root) / "scratch",
                )
            else:
                event_receipt = _persist_event_dispatch_v1(
                    runtime,
                    dispatch,
                    scratch_root=Path(context.batch_root) / "scratch",
                    receipt_factory=provider._sink_receipt_factory,
                )
            compact_phase_timing[
                "direct_sparse_durable_publication_seconds"
            ] += (perf_counter() - durable_phase_started)
            event_receipts.append(event_receipt)
            if direct_sparse:
                inventory_phase_started = perf_counter()
                record_direct_sparse_artifacts_v1(event_receipt)
                compact_phase_timing[
                    "direct_sparse_lifecycle_inventory_seconds"
                ] += (perf_counter() - inventory_phase_started)
            if compact_science_capture_enabled:
                if compact_science_capture is None:
                    raise CampaignProviderError(
                        "compact science capture did not return its detached boundary"
                    )
                if compact_publication_pipeline is None:
                    raise CampaignProviderError(
                        "compact science async publication pipeline is absent"
                    )
                destination = (
                    Path(context.batch_root)
                    / "science-compact"
                    / "dispatches"
                    / (
                        f"ticks-{current:012d}-"
                        f"{current + dispatch_ticks:012d}"
                    )
                )
                freeze_phase_started = perf_counter()
                try:
                    frozen_compact_dispatch = (
                        _compact_science_capture
                        .freeze_compact_science_dispatch_host_v1(
                            dispatch=dispatch,
                            capture=compact_science_capture,
                            world_cells=science_cells,
                            native_world_control_set_sha256=str(
                                native_world_control_set_sha256
                            ),
                            source_sink_receipt_sha256=str(
                                event_receipt["sink_receipt_sha256"]
                            ),
                        )
                    )
                    compact_publication_pipeline.submit(
                        frozen=frozen_compact_dispatch,
                        destination=destination,
                    )
                except Exception as exc:
                    raise CampaignProviderError(
                        f"compact science dispatch admission failed: {exc}"
                    ) from exc
                finally:
                    compact_phase_timing[
                        "compact_host_freeze_and_submit_seconds"
                    ] += (perf_counter() - freeze_phase_started)
                pending_compact_event_receipts.append(event_receipt)
            if science_capture_enabled:
                if state_before is None or science_capture is None:
                    raise CampaignProviderError(
                        "typed science capture did not return its detached boundary"
                    )
                state_after = runtime.state_projection()
                from engine.native_runtime.cuda_v2_science_capture_v1 import (
                    build_native_cuda_v2_observation_outputs_v1,
                )

                outputs = build_native_cuda_v2_observation_outputs_v1(
                    dispatch=dispatch,
                    capture=science_capture,
                    state_before=state_before,
                    state_after=state_after,
                    world_cells=science_cells,
                    applied_native_control=native_control,
                    emit_control_binding=not any(science_records_by_cell.values()),
                )
                _science_append_output_v1(
                    outputs=outputs,
                    records_by_cell=science_records_by_cell,
                    cells=science_cells,
                )
                for cell_id, rows in science_records_by_cell.items():
                    for row in reversed(rows):
                        if (
                            row.record_kind.value == "world_state_snapshot"
                            and row.source_state_after_sha256 is not None
                        ):
                            science_state_sha_by_cell[cell_id] = (
                                row.source_state_after_sha256.to_dict()
                            )
                            break
            current += dispatch_ticks
            if request.total_ticks is not None:
                progress_recorder = getattr(context, "record_progress", None)
                if callable(progress_recorder):
                    progress_recorder(current, stage="native dispatch")
            settle_ready_compact_publications_v1()
            if (
                compact_science_capture_enabled
                and current in compact_snapshot_ticks
            ):
                persist_compact_snapshot_v1(absolute_tick=current)
            settle_ready_acs_checkpoints()
            if current in checkpoint_ticks:
                compact_science_continuation = (
                    persist_compact_checkpoint_frontier_v1(
                        absolute_tick=current
                    )
                )
                if acs_pipeline is None:
                    sealed = (
                        runtime.checkpoint()
                        if direct_sparse
                        else runtime.checkpoint(to_cpu=True)
                    )
                    bundle, checkpoint_id, tree_digest, state_sha256 = (
                        _publish_checkpoint_bundle_v1(
                            sealed,
                            context=context,
                            artifact=artifact,
                            occurrence_adapter=occurrence_adapter,
                        )
                    )
                else:
                    from .cuda_full_evolution_v2_acs_adapter import (
                        CudaFullEvolutionV2CapturePoolV1,
                        capture_native_checkpoint_to_file_v1,
                        capture_native_resident_checkpoint_to_file_v1,
                    )

                    checkpoint_root = Path(context.batch_root) / "checkpoints"
                    checkpoint_root.mkdir(parents=True, exist_ok=True)
                    checkpoint_destination = checkpoint_root / (
                        f"checkpoint-tick-{current:012d}"
                    )
                    while len(pending_acs_checkpoints) >= acs_pipeline.slots:
                        settle_oldest_acs_checkpoint(producer_backpressure=True)
                    token = acs_pipeline.begin_capture(compute_frontier=current)
                    try:
                        source_factory = getattr(
                            runtime, "checkpoint_capture_source_v1", None
                        )
                        if callable(source_factory):
                            capture_mode = "direct_resident_cuda_rows_v1"
                            if acs_capture_mode not in {None, capture_mode}:
                                raise CampaignProviderError(
                                    "CUDA Native V2 ACS capture mode changed within one "
                                    "resident-owner execution"
                                )
                            acs_capture_mode = capture_mode
                            source = source_factory()
                            if acs_capture_pool is None:
                                acs_capture_pool = CudaFullEvolutionV2CapturePoolV1(
                                    has_cuda_fields=True,
                                    require_native_pool=(
                                        provider._acs_require_native_pool
                                    ),
                                )
                            capture = capture_native_resident_checkpoint_to_file_v1(
                                source,
                                checkpoint_destination,
                                artifact_path=artifact["binary_path"],
                                checkpoint_at_tick=current,
                                compact_science_frontier=(
                                    compact_science_continuation
                                ),
                                require_native_pool=provider._acs_require_native_pool,
                                capture_pool=acs_capture_pool,
                            )
                            del source
                        else:
                            if provider._acs_require_direct_resident_capture:
                                raise CampaignProviderError(
                                    "CUDA Native V2 ACS runtime lacks direct resident "
                                    "checkpoint capture"
                                )
                            sealed = (
                                runtime.checkpoint()
                                if direct_sparse
                                else runtime.checkpoint(to_cpu=True)
                            )
                            capture_mode = "synchronous_checkpoint_host_spool_v1"
                            if acs_capture_mode not in {None, capture_mode}:
                                raise CampaignProviderError(
                                    "CUDA Native V2 ACS capture mode changed within one "
                                    "resident-owner execution"
                                )
                            acs_capture_mode = capture_mode
                            runtime_checkpoint = (
                                _validated_checkpoint_runtime_projection_v1(
                                    sealed,
                                    artifact=artifact,
                                )
                            )
                            if acs_capture_pool is None:
                                acs_capture_pool = CudaFullEvolutionV2CapturePoolV1(
                                    has_cuda_fields=False,
                                    require_native_pool=(
                                        provider._acs_require_native_pool
                                    ),
                                )
                            capture = capture_native_checkpoint_to_file_v1(
                                sealed,
                                checkpoint_destination,
                                runtime_checkpoint=runtime_checkpoint,
                                artifact_path=artifact["binary_path"],
                                checkpoint_at_tick=current,
                                compact_science_frontier=(
                                    compact_science_continuation
                                ),
                                require_native_pool=provider._acs_require_native_pool,
                                capture_pool=acs_capture_pool,
                            )
                            del sealed
                        acs_pipeline.submit_capture(token, capture)
                    except BaseException as exc:
                        acs_pipeline.fail_capture(token, exc)
                        raise
                    pending_acs_checkpoints.append(
                        {
                            "token": token,
                            "checkpoint_at_tick": current,
                            "compact_science_frontier": (
                                compact_science_continuation
                            ),
                        }
                    )
                if acs_pipeline is None:
                    occurrence_root: Path | None = None
                    try:
                        occurrence_digest, occurrence_root = (
                            _materialize_authoritative_occurrence_v1(
                                bundle,
                                checkpoint_id=checkpoint_id,
                                tree_digest=tree_digest,
                                absolute_next_tick=current,
                                context=context,
                                source_batch_id=request.batch_id,
                                attempt_authority=attempt_authority,
                                parent_checkpoint_digest=parent_occurrence_digest,
                                occurrence_adapter=occurrence_adapter,
                            )
                        )
                        science_observation_frontier = (
                            append_science_checkpoint_marker_v1(
                                checkpoint_occurrence_id=(
                                    str(occurrence_digest["hex"])
                                    if occurrence_digest is not None
                                    else checkpoint_id
                                ),
                                absolute_tick=current,
                                state_sha256=state_sha256,
                            )
                        )
                        latest_checkpoint = _checkpoint_record_v1(
                            request=request,
                            context=context,
                            bundle=bundle,
                            checkpoint_id=checkpoint_id,
                            tree_digest=tree_digest,
                            state_sha256=state_sha256,
                            absolute_next_tick=current,
                            occurrence_digest=occurrence_digest,
                            occurrence_root=occurrence_root,
                            completed_restore_ticks=completed_restore_ticks,
                            science_continuation_admission=(
                                _base_checkpoint_science_admission_v1(artifact)
                            ),
                            science_observation_frontier=science_observation_frontier,
                            compact_science_frontier=(
                                compact_science_continuation
                            ),
                            native_control=native_control,
                        )
                        context.record_checkpoint(latest_checkpoint)
                    except BaseException as exc:
                        _quarantine_after_external_commit_failure_v1(
                            occurrence_root,
                            exc,
                            label="checkpoint occurrence",
                        )
                        _quarantine_after_external_commit_failure_v1(
                            bundle,
                            exc,
                            label="checkpoint bundle",
                        )
                        raise
                    if occurrence_digest is not None:
                        parent_occurrence_digest = occurrence_digest
                if (
                    current in request.deliberate_restore_ticks
                    and current not in completed_restore_ticks
                ):
                    if acs_pipeline is not None:
                        runtime.close()
                        runtime = None
                        settle_all_acs_checkpoints(producer_backpressure=True)
                    if (
                        latest_checkpoint is None
                        or latest_checkpoint.get("occurrence_digest") is None
                    ):
                        raise CampaignProviderError(
                            "deliberate restore requires Scheduler-owned CRCA attempt authority"
                        )
                    if acs_pipeline is not None:
                        acs_pipeline.close()
                    close_compact_publication_v1()
                    close_native_snapshot_publication_v1()
                    return {
                        "status": "interrupted",
                        "checkpoint": latest_checkpoint,
                        "metrics_rows": [],
                        "result": {
                            "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                            "next_absolute_tick": current,
                            "crca_final_frontiers": {
                                "last_dispatched_next_tick": current,
                                "last_computed_next_tick": current,
                                "durable_checkpoint_next_tick": current,
                                "published_artifact_next_tick": current,
                            },
                            "event_dispatch_receipts": event_receipts,
                            "native_control": native_control,
                            "science_continuation_admission": (
                                _base_checkpoint_science_admission_v1(artifact)
                            ),
                            "science_observation_frontier": (
                                latest_checkpoint.get("science_observation_frontier")
                                if science_capture_enabled
                                else None
                            ),
                            "compact_science_frontier": (
                                latest_checkpoint.get("compact_science_frontier")
                                if compact_science_capture_enabled
                                else None
                            ),
                            "compact_science_async_publication": (
                                compact_publication_telemetry
                            ),
                            **(
                                {
                                    "compact_science_native_snapshot_publication": (
                                        native_snapshot_publication_telemetry
                                    )
                                }
                                if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
                                    compact_snapshot_implementation
                                )
                                else {}
                            ),
                            "technical_only": True,
                            "formal_qualification_authority": False,
                            "acs_checkpoint_staging": {
                                **acs_pipeline.telemetry(),
                                "capture_mode": acs_capture_mode,
                                "checkpoint_receipts": list(acs_checkpoint_receipts),
                            }
                            if acs_pipeline is not None
                            else None,
                        },
                    }
        if acs_pipeline is not None:
            if direct_sparse:
                direct_sparse_execution_receipt = dict(
                    runtime.execution_receipt_v3()
                )
            counters = runtime.counters()
            runtime.close()
            runtime = None
            settle_all_acs_checkpoints(producer_backpressure=True)
        if (
            latest_checkpoint is None
            or latest_checkpoint["next_absolute_tick"] != target
        ):
            raise CampaignProviderError("CUDA Native V2 terminal checkpoint is absent")
        if acs_pipeline is not None:
            acs_pipeline.close()
        close_compact_publication_v1()
        # Terminal capture indexes are authoritative over their ordered
        # pointers; settle native work before constructing or recording one.
        close_native_snapshot_publication_v1()
        if science_capture_enabled:
            from engine.native_runtime.cuda_v2_science_capture_v1 import (
                build_native_cuda_v2_completion_output_v1,
            )
            for cell_id in sorted(science_records_by_cell):
                rows = science_records_by_cell[cell_id]
                control_binding_digest = _science_control_binding_digest_v1(
                    rows,
                    expected_native_control=native_control,
                )
                prefix = canonical_science_record_stream_sha256_v1(tuple(rows)).to_dict()
                terminal_state = science_state_sha_by_cell.get(
                    cell_id,
                    {"algorithm": "sha256", "hex": latest_checkpoint["state_sha256"]},
                )
                completion_output = build_native_cuda_v2_completion_output_v1(
                    cell_id=cell_id,
                    absolute_tick=target,
                    completion_class="TERMINAL_HORIZON",
                    terminal_state_sha256=terminal_state,
                    terminal_population_count=_science_population_count_v1(rows),
                    terminal_record_count=len(rows),
                    canonical_record_stream_sha256=prefix,
                    checkpoint_occurrence_ids=tuple(science_checkpoint_occurrences),
                    control_binding_sha256=control_binding_digest,
                )
                completion_records = (
                    _science_observation_projection
                    .finalize_native_cuda_v2_observations_v1(
                        completion_output,
                        context=_science_context_v1(science_cells[cell_id]),
                        start_world_local_sequence=len(rows),
                    )
                )
                science_records_by_cell[cell_id].extend(completion_records)
            science_observation_frontier = _science_sidecar_snapshot_v1(
                context=context,
                request=request,
                records_by_cell=science_records_by_cell,
                next_absolute_tick=target,
                terminal=True,
            )
            latest_checkpoint = {
                **latest_checkpoint,
                "science_observation_frontier": science_observation_frontier,
            }
            # Rebind the lifecycle checkpoint projection to the terminal
            # sidecar frontier.  The canonical base checkpoint tree remains
            # unchanged; only the additive host-side observation pointer moves.
            context.record_checkpoint(latest_checkpoint)
        if compact_science_capture_enabled:
            compact_capture_index = _compact_science_index_v1(
                context=context,
                request=request,
                native_world_control_set_sha256=str(
                    native_world_control_set_sha256
                ),
                dispatch_receipts=compact_dispatch_receipts,
                snapshot_receipts=compact_snapshot_receipts,
                checkpoint_frontier=compact_checkpoint_frontier,
                terminal_tick=target,
            )
            latest_checkpoint = {
                **latest_checkpoint,
                "compact_science_capture_index": dict(compact_capture_index),
            }
            context.record_checkpoint(latest_checkpoint)
        direct_sparse_execution_receipt = (
            direct_sparse_execution_receipt
            if acs_pipeline is not None
            else (runtime.execution_receipt_v3() if direct_sparse else None)
        )
        _validate_native_control_runtime_receipt_v1(
            direct_sparse_execution_receipt, native_control
        )
        _validate_native_world_control_runtime_receipt_v1(
            direct_sparse_execution_receipt,
            expected_controls=native_world_controls,
            expected_control_set_sha256=native_world_control_set_sha256,
        )
        if acs_pipeline is None:
            counters = runtime.counters()
        return {
            "status": "completed",
            "checkpoint": latest_checkpoint,
            "metrics_rows": [],
            "result": {
                "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
                "next_absolute_tick": current,
                "crca_final_frontiers": {
                    "last_dispatched_next_tick": current,
                    "last_computed_next_tick": current,
                    "durable_checkpoint_next_tick": current,
                    "published_artifact_next_tick": current,
                },
                "world_count": contract.B,
                "committed_world_ticks": contract.B * contract.total_ticks,
                "capability_counters": counters,
                "event_dispatch_receipts": event_receipts,
                "execution_mode": execution_mode,
                "event_transport": event_transport,
                "horizon_dependent_resident_bytes": 0 if direct_sparse else None,
                "science_continuation_admission": (
                    _base_checkpoint_science_admission_v1(artifact)
                ),
                "direct_sparse_execution_receipt": direct_sparse_execution_receipt,
                "native_control": native_control,
                "science_observation_frontier": (
                    latest_checkpoint.get("science_observation_frontier")
                    if science_capture_enabled
                    else None
                ),
                "science_observation_record_counts": (
                    {
                        cell_id: len(rows)
                        for cell_id, rows in sorted(science_records_by_cell.items())
                    }
                    if science_capture_enabled
                    else None
                ),
                "compact_science_capture_index": compact_capture_index,
                "compact_science_dispatch_artifact_count": (
                    len(compact_dispatch_receipts)
                    if compact_science_capture_enabled
                    else None
                ),
                "compact_science_snapshot_artifact_count": (
                    len(compact_snapshot_receipts)
                    if compact_science_capture_enabled
                    else None
                ),
                "compact_science_async_publication": (
                    compact_publication_telemetry
                    if compact_science_capture_enabled
                    else None
                ),
                **(
                    {
                        "compact_science_native_snapshot_publication": (
                            native_snapshot_publication_telemetry
                        )
                    }
                    if _native_snapshot_publication.native_snapshot_implementation_is_native_v1(
                        compact_snapshot_implementation
                    )
                    else {}
                ),
                **_compact_snapshot_timing_result_fields_v1(
                    enabled=compact_snapshot_timing_enabled,
                    samples=compact_snapshot_timing_samples,
                ),
                "device_identity": selection["device_identity"],
                "artifact_sha256": artifact["binary_sha256"],
                "manifest_sha256": artifact["manifest_sha256"],
                "technical_only": True,
                "formal_qualification_authority": False,
                "physical_batch_authorization": (
                    None
                    if physical_batch_authorization is None
                    else dict(physical_batch_authorization)
                ),
                "acs_checkpoint_staging": {
                    **acs_pipeline.telemetry(),
                    "capture_mode": acs_capture_mode,
                    "checkpoint_receipts": list(acs_checkpoint_receipts),
                }
                if acs_pipeline is not None
                else None,
            },
        }
    except BaseException as exc:
        execution_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        # Preserve the capture-fence lifetime: an abort first waits/cancels the
        # bounded native worker while its bound runtime owner and CUDA events
        # are still alive.  Only then may runtime.close release resident state.
        if native_snapshot_pipeline is not None:
            try:
                telemetry = native_snapshot_pipeline.telemetry()
                if not telemetry.get("closed"):
                    native_snapshot_pipeline.abort()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            finally:
                runtime = None
        if compact_publication_pipeline is not None:
            try:
                telemetry = compact_publication_pipeline.telemetry()
                if not telemetry.get("closed"):
                    compact_publication_pipeline.abort()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if acs_pipeline is not None:
            try:
                telemetry = acs_pipeline.telemetry()
                if not telemetry.get("closed"):
                    acs_pipeline.abort()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if execution_error is not None:
                for cleanup_error in cleanup_errors:
                    execution_error.add_note(
                        "CUDA Native V2 cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            else:
                first_cleanup_error = cleanup_errors[0]
                for cleanup_error in cleanup_errors[1:]:
                    first_cleanup_error.add_note(
                        "additional CUDA Native V2 cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise first_cleanup_error


def _reject_generation_authority_v1(value: object, *, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CampaignProviderError(f"{path} contains a non-string key")
            lowered = key.lower()
            if "generation" in lowered:
                raise CampaignProviderError(
                    f"{path}.{key} is generation-shaped; use absolute ticks"
                )
            _reject_generation_authority_v1(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_generation_authority_v1(child, path=f"{path}[{index}]")


def _autotune_selection_v1(
    receipt_path: Path,
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: Mapping[str, Any],
    event_transport: str = DENSE_EVENT_TRANSPORT_V2,
) -> tuple[int, dict[str, Any]]:
    path = _regular_unlinked_file(receipt_path, name="CUDA Native V2 tuning receipt")
    receipt = _json_object(path, name="CUDA Native V2 tuning receipt")
    if event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3:
        return _direct_sparse_autotune_selection_v3(
            path,
            receipt,
            artifact_sha256=artifact_sha256,
            device_identity=device_identity,
        )
    if event_transport != DENSE_EVENT_TRANSPORT_V2:
        raise CampaignProviderError("tuning receipt event transport is invalid")
    if (
        receipt.get("schema_version") != "eiel_full_evolution_cuda_event_autotune_v1"
        or receipt.get("route_id") != CUDA_NATIVE_ROUTE_V2
    ):
        raise CampaignProviderError("tuning receipt schema/route is invalid")
    artifact = receipt.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("sha256") != artifact_sha256
        or artifact.get("manifest_sha256") != manifest_sha256
    ):
        raise CampaignProviderError("tuning receipt binds another artifact")
    if receipt.get("device_identity") != dict(device_identity):
        raise CampaignProviderError("tuning receipt binds another CUDA device")
    selected = receipt.get("selected")
    candidate = selected.get("candidate") if isinstance(selected, Mapping) else None
    if (
        not isinstance(selected, Mapping)
        or selected.get("status") != "VALID"
        or not isinstance(candidate, Mapping)
        or type(candidate.get("temporal_k")) is not int
        or not TEMPORAL_K_MIN_V2 <= candidate["temporal_k"] <= TEMPORAL_K_MAX_V2
    ):
        raise CampaignProviderError("tuning receipt lacks a valid selected K")
    return int(candidate["temporal_k"]), receipt


def _direct_sparse_autotune_selection_v3(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    *,
    artifact_sha256: str,
    device_identity: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    expected_fields = {
        "schema_version",
        "claim_class",
        "selection_statistic",
        "artifact",
        "device_identity",
        "plan",
        "probe_samples",
        "probe_warmups",
        "measurements",
        "selected",
        "autotune_source",
        "result_sha256",
    }
    if set(receipt) != expected_fields or (
        receipt.get("schema_version") != DIRECT_SPARSE_AUTOTUNE_SCHEMA_V3
        or receipt.get("claim_class") != "TECHNICAL_ONLY"
        or receipt.get("selection_statistic")
        != "p10_then_minimum_committed_world_ticks_per_second"
    ):
        raise CampaignProviderError("direct-sparse tuning receipt shape is invalid")
    supplied = _sha256_text(
        receipt.get("result_sha256"), name="direct-sparse tuning receipt digest"
    )
    if supplied != canonical_sha256(
        {key: value for key, value in receipt.items() if key != "result_sha256"}
    ):
        raise CampaignProviderError("direct-sparse tuning receipt digest mismatch")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
        raise CampaignProviderError("direct-sparse tuning artifact binding is invalid")
    tuned_artifact = _regular_unlinked_file(
        Path(str(artifact.get("path", ""))).expanduser(),
        name="direct-sparse tuned artifact",
    )
    if (
        artifact.get("sha256") != artifact_sha256
        or _sha256_file(tuned_artifact) != artifact_sha256
    ):
        raise CampaignProviderError("direct-sparse tuning binds another artifact")
    if receipt.get("device_identity") != dict(device_identity):
        raise CampaignProviderError("direct-sparse tuning binds another CUDA device")
    plan = receipt.get("plan")
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("direct-sparse tuning plan is absent")
    device_envelope = plan.get("device_envelope")
    scaling_law = plan.get("scaling_law")
    plan_core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        set(plan)
        != {
            "schema_version",
            "claim_class",
            "device_envelope",
            "scaling_law",
            "temporal_k_candidates",
            "candidates",
            "plan_sha256",
        }
        or plan.get("schema_version")
        != "eiel_cuda_full_evolution_direct_sparse_autotune_plan_v3"
        or plan.get("claim_class") != "TECHNICAL_ONLY"
        or _sha256_text(
            plan.get("plan_sha256"), name="direct-sparse tuning plan digest"
        )
        != canonical_sha256(plan_core)
        or not isinstance(plan.get("candidates"), list)
        or not plan["candidates"]
        or not isinstance(device_envelope, Mapping)
        or set(device_envelope)
        != {"multiprocessor_count", "current_free_bytes", "total_memory_bytes"}
        or device_envelope.get("total_memory_bytes")
        != device_identity.get("total_memory_bytes")
        or not isinstance(scaling_law, Mapping)
        or scaling_law.get("horizon_dependent_resident_bytes") != 0
    ):
        raise CampaignProviderError("direct-sparse tuning plan binding is invalid")
    selected = receipt.get("selected")
    measurements = receipt.get("measurements")
    candidate = selected.get("candidate") if isinstance(selected, Mapping) else None
    measurement_fields = {
        "candidate",
        "status",
        "minimum_committed_world_ticks_per_second",
        "p10_committed_world_ticks_per_second",
        "result_path",
        "result_body_sha256",
    }
    measurements_are_exact = isinstance(measurements, list) and bool(measurements)
    if measurements_are_exact:
        for row in measurements:
            if not isinstance(row, Mapping) or set(row) != measurement_fields:
                measurements_are_exact = False
                break
            row_candidate = row.get("candidate")
            minimum = row.get("minimum_committed_world_ticks_per_second")
            p10 = row.get("p10_committed_world_ticks_per_second")
            if (
                not isinstance(row_candidate, Mapping)
                or dict(row_candidate) not in plan["candidates"]
                or type(minimum) not in {int, float}
                or type(p10) not in {int, float}
                or not math.isfinite(float(minimum))
                or not math.isfinite(float(p10))
                or float(minimum) <= 0.0
                or float(p10) <= 0.0
                or not isinstance(row.get("status"), str)
                or not row["status"]
            ):
                measurements_are_exact = False
                break
    if (
        not isinstance(selected, Mapping)
        or set(selected) != measurement_fields
        or not measurements_are_exact
        or dict(selected) not in measurements
        or dict(selected)
        != max(
            measurements,
            key=lambda row: (
                row["p10_committed_world_ticks_per_second"],
                row["minimum_committed_world_ticks_per_second"],
            ),
        )
        or not isinstance(candidate, Mapping)
        or dict(candidate) not in plan["candidates"]
        or set(candidate)
        != {
            "batch_size",
            "temporal_k",
            "initial_capacity_rows_per_world",
            "estimated_peak_bytes",
            "blocks_per_sm_wave_target",
        }
        or type(candidate.get("batch_size")) is not int
        or candidate["batch_size"] <= 0
        or type(candidate.get("temporal_k")) is not int
        or not TEMPORAL_K_MIN_V2 <= candidate["temporal_k"] <= TEMPORAL_K_MAX_V2
        or type(candidate.get("initial_capacity_rows_per_world")) is not int
        or candidate["initial_capacity_rows_per_world"] <= 0
        or type(candidate.get("estimated_peak_bytes")) is not int
        or candidate["estimated_peak_bytes"] <= 0
        or not isinstance(selected.get("status"), str)
    ):
        raise CampaignProviderError(
            "direct-sparse tuning lacks one exact selected candidate"
        )
    result_path = _regular_unlinked_file(
        Path(str(selected.get("result_path", ""))).expanduser(),
        name="direct-sparse selected benchmark result",
    )
    result = _json_object(result_path, name="direct-sparse selected benchmark result")
    result_body_sha256 = _sha256_text(
        selected.get("result_body_sha256"),
        name="direct-sparse selected benchmark body digest",
    )
    result_artifact = result.get("artifact")
    result_candidate = result.get("candidate")
    resource_admission = result.get("resource_admission")
    minimum_throughput = result.get("minimum_committed_world_ticks_per_second")
    p10_throughput = result.get("p10_committed_world_ticks_per_second")
    if (
        result.get("schema_version")
        != "eiel_cuda_full_evolution_direct_sparse_sustained_benchmark_v2"
        or result.get("result_body_sha256") != result_body_sha256
        or canonical_sha256(
            {key: value for key, value in result.items() if key != "result_body_sha256"}
        )
        != result_body_sha256
        or not isinstance(result_artifact, Mapping)
        or result_artifact.get("sha256") != artifact_sha256
        or result.get("device_identity") != dict(device_identity)
        or not isinstance(result_candidate, Mapping)
        or result_candidate.get("batch_size") != candidate["batch_size"]
        or result_candidate.get("temporal_k") != candidate["temporal_k"]
        or result_candidate.get("initial_capacity_rows_per_world")
        != candidate["initial_capacity_rows_per_world"]
        or result.get("horizon_dependent_resident_bytes") != 0
        or result.get("status") != selected.get("status")
        or minimum_throughput
        != selected.get("minimum_committed_world_ticks_per_second")
        or p10_throughput != selected.get("p10_committed_world_ticks_per_second")
        or type(minimum_throughput) not in {int, float}
        or type(p10_throughput) not in {int, float}
        or not math.isfinite(float(minimum_throughput))
        or not math.isfinite(float(p10_throughput))
        or result.get("target_world_ticks_per_second")
        != DIRECT_SPARSE_MINIMUM_COMMITTED_WORLD_TICKS_PER_SECOND_V3
        or not isinstance(resource_admission, Mapping)
        or resource_admission.get("required_free_bytes")
        != candidate["estimated_peak_bytes"]
        or resource_admission.get("total_device_bytes")
        != device_identity.get("total_memory_bytes")
        or resource_admission.get("horizon_dependent_resident_bytes") != 0
    ):
        raise CampaignProviderError(
            "direct-sparse selected benchmark binding is invalid"
        )
    normalized = dict(receipt)
    normalized["_provider_selection"] = {
        "batch_size": int(candidate["batch_size"]),
        "temporal_k": int(candidate["temporal_k"]),
        "initial_capacity_rows_per_world": int(
            candidate["initial_capacity_rows_per_world"]
        ),
        "estimated_peak_bytes": int(candidate["estimated_peak_bytes"]),
        "result_body_sha256": result_body_sha256,
        "result_file_sha256": _sha256_file(result_path),
        "autotune_result_sha256": supplied,
        "benchmark_status": str(result["status"]),
        "minimum_committed_world_ticks_per_second": float(minimum_throughput),
        "p10_committed_world_ticks_per_second": float(p10_throughput),
        "target_world_ticks_per_second": (
            DIRECT_SPARSE_MINIMUM_COMMITTED_WORLD_TICKS_PER_SECOND_V3
        ),
    }
    return int(candidate["temporal_k"]), normalized


def _load_yaml_object_v1(path: Path) -> dict[str, Any]:
    source = _regular_unlinked_file(path, name="CUDA Native V2 campaign template")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CampaignProviderError(f"cannot load campaign template: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CampaignProviderError("campaign template must be a YAML object")
    return dict(value)


def _selection_receipt_v1(
    *,
    artifact_sha256: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
    device_identity: Mapping[str, Any],
    tuning_path: Path,
    tuning_file_sha256: str,
    tuning_schema: str,
    temporal_k: int,
    execution_mode: str,
    requested_batch_size: int,
    selected_batch_size: int,
    reserve_bytes: int,
    required_bytes: int,
    maximum_working_bytes: int,
    headroom_bytes: int,
    created_at_utc: str,
    event_transport: str = DENSE_EVENT_TRANSPORT_V2,
    initial_capacity_rows_per_world: int | None = None,
    autotune_result_sha256: str | None = None,
    selected_benchmark_body_sha256: str | None = None,
    selected_benchmark_file_sha256: str | None = None,
) -> dict[str, Any]:
    if event_transport not in EVENT_TRANSPORTS_V2:
        raise CampaignProviderError("selection event transport is invalid")
    direct_sparse = event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
    if direct_sparse and (
        type(initial_capacity_rows_per_world) is not int
        or initial_capacity_rows_per_world <= 0
        or autotune_result_sha256 is None
        or selected_benchmark_body_sha256 is None
        or selected_benchmark_file_sha256 is None
    ):
        raise CampaignProviderError(
            "direct-sparse selection lacks exact autotune resource bindings"
        )
    core = {
        "schema_version": (
            DEVICE_SELECTION_SCHEMA_V2 if direct_sparse else DEVICE_SELECTION_SCHEMA_V1
        ),
        "profile_id": PROFILE_ID,
        "provider_id": PROVIDER_ID,
        "route_id": CUDA_NATIVE_ROUTE_V2,
        "native_abi": CUDA_NATIVE_ABI_V1,
        "artifact": {
            "binary_sha256": artifact_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest_sha256,
        },
        "device_identity": dict(device_identity),
        "capacity_basis": {
            "total_vram_bytes": int(device_identity["total_memory_bytes"]),
            "fixed_reserve_bytes": reserve_bytes,
            "working_fraction_numerator": 3,
            "working_fraction_denominator": 4,
            "transient_free_vram_used": False,
        },
        "tuning_binding": {
            "path": str(tuning_path),
            "file_sha256": tuning_file_sha256,
            "schema_version": tuning_schema,
            "temporal_k": temporal_k,
            "execution_mode": execution_mode,
            **(
                {
                    "event_transport": event_transport,
                    "autotune_result_sha256": _sha256_text(
                        autotune_result_sha256,
                        name="direct-sparse autotune result digest",
                    ),
                    "selected_benchmark_body_sha256": _sha256_text(
                        selected_benchmark_body_sha256,
                        name="direct-sparse selected benchmark body digest",
                    ),
                    "selected_benchmark_file_sha256": _sha256_text(
                        selected_benchmark_file_sha256,
                        name="direct-sparse selected benchmark file digest",
                    ),
                }
                if direct_sparse
                else {}
            ),
        },
        "selection": {
            "requested_batch_size": requested_batch_size,
            "selected_batch_size": selected_batch_size,
            "temporal_k": temporal_k,
            "execution_mode": execution_mode,
            **(
                {
                    "event_transport": event_transport,
                    "initial_capacity_rows_per_world": initial_capacity_rows_per_world,
                    "horizon_dependent_resident_bytes": 0,
                }
                if direct_sparse
                else {}
            ),
            "required_bytes": required_bytes,
            "maximum_working_bytes": maximum_working_bytes,
            "headroom_bytes": headroom_bytes,
            "admitted": True,
        },
        "created_at_utc": created_at_utc,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def concretize_device_spec_and_request_v1(
    template_path: str | Path,
    *,
    artifact_path: str | Path,
    acs_native_runtime_extension_path: str | Path | None = None,
    tuning_receipt_path: str | Path,
    output_spec_path: str | Path,
    output_request_path: str | Path,
    output_root: str | Path,
    requested_batch_size: int | None = None,
    reserve_bytes: int = DEFAULT_FIXED_VRAM_RESERVE_BYTES_V1,
    purpose: str = "qualprep",
    event_transport: str | None = None,
    acs_checkpoint_slots: int = 0,
    gpu_resource_policy: str | None = None,
    logical_world_count: int | None = None,
    device: str = "cuda",
    created_at_utc: str | None = None,
    artifact_validator: ArtifactValidatorV1 = validate_prebuilt_cuda_full_evolution_v2,
    artifact_loader: ArtifactLoaderV1 = load_prebuilt_cuda_full_evolution_v2,
) -> dict[str, Any]:
    """Seal live-device/B/K inputs before campaign compilation.

    The only device-memory capacity input is the selected device's immutable
    total VRAM identity.  Transient free VRAM is intentionally never read.
    """

    template = _load_yaml_object_v1(Path(template_path).expanduser())
    if not is_cuda_full_evolution_v2_spec_v1(template):
        raise CampaignProviderError("campaign template is not the CUDA Native V2 world")
    if "run_contract" in template:
        raise CampaignProviderError(
            "CUDA Native V2 template must not contain run_contract"
        )
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise CampaignProviderError("fixed VRAM reserve must be nonnegative")
    if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
        raise CampaignProviderError("ACS checkpoint slots must be zero, one, or two")
    acs_native_extension: Path | None = None
    acs_native_manifest: Path | None = None
    if acs_checkpoint_slots == 0:
        if acs_native_runtime_extension_path is not None:
            raise CampaignProviderError(
                "ACS native runtime extension requires one or two ACS slots"
            )
    else:
        if acs_native_runtime_extension_path is None:
            raise CampaignProviderError(
                "ACS1/ACS2 requires an exact AOT native runtime extension"
            )
        declared_extension = Path(
            acs_native_runtime_extension_path
        ).expanduser()
        acs_native_extension = _regular_unlinked_file(
            declared_extension,
            name="ACS native runtime extension",
        ).resolve()
        acs_native_manifest = _regular_unlinked_file(
            acs_native_extension.with_suffix(
                acs_native_extension.suffix + ".manifest.json"
            ),
            name="ACS native runtime extension manifest",
        ).resolve()
    shared_w32 = gpu_resource_policy == SHARED_W32_GPU_POLICY_ID_V1
    if gpu_resource_policy is not None and not shared_w32:
        raise CampaignProviderError(
            "GPU resource policy must select the exact shared W32 policy"
        )
    if logical_world_count is not None and (
        type(logical_world_count) is not int or logical_world_count <= 0
    ):
        raise CampaignProviderError("logical world count must be a positive integer")
    if not shared_w32 and logical_world_count is not None:
        raise CampaignProviderError(
            "logical world count is independently selectable only for shared W32"
        )
    if shared_w32 and purpose == "qualification":
        raise CampaignProviderError(
            "shared W32 is a technical-only development policy, not Qualification"
        )
    if shared_w32 and acs_checkpoint_slots > SHARED_W32_MAX_ACS_SLOTS_V1:
        raise CampaignProviderError("shared W32 permits at most ACS1")
    artifact, manifest_path, manifest = artifact_validator(artifact_path)
    artifact = artifact.resolve()
    manifest_path = manifest_path.resolve()
    artifact_sha256 = _sha256_file(artifact)
    manifest_file_sha256 = _sha256_file(manifest_path)
    manifest_sha256 = _sha256_text(
        manifest.get("manifest_sha256"), name="native manifest identity"
    )
    loaded = artifact_loader(artifact, device=device)
    if (
        str(getattr(loaded, "artifact_sha256", "")) != artifact_sha256
        or str(getattr(loaded, "manifest_sha256", "")) != manifest_sha256
    ):
        raise CampaignProviderError(
            "loaded artifact identity differs during concretization"
        )
    device_identity = dict(getattr(loaded, "device_identity", {}))
    total_vram = device_identity.get("total_memory_bytes")
    if type(total_vram) is not int or total_vram <= 0:
        raise CampaignProviderError("loaded device lacks exact total VRAM identity")
    tuning_path = _regular_unlinked_file(
        Path(tuning_receipt_path).expanduser(), name="CUDA Native V2 tuning receipt"
    )
    parameters = template.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CampaignProviderError("template parameters are invalid")
    execution_mode = parameters.get("cuda_native.execution_mode")
    if execution_mode not in EXECUTION_MODES_V2:
        raise CampaignProviderError(
            "template must explicitly select a supported CUDA execution mode"
        )
    event_transport = event_transport or parameters.get(
        "cuda_native.event_transport", DENSE_EVENT_TRANSPORT_V2
    )
    if event_transport not in EVENT_TRANSPORTS_V2:
        raise CampaignProviderError(
            "template must explicitly select a supported event transport"
        )
    if shared_w32 and event_transport != DIRECT_SPARSE_EVENT_TRANSPORT_V3:
        raise CampaignProviderError("shared W32 requires direct_sparse_v3")
    science_observation_mode = parameters.get("cuda_native.science_observation_mode")
    if science_observation_mode not in SCIENCE_OBSERVATION_MODES_V1:
        raise CampaignProviderError(
            "template must explicitly bind a supported science observation mode"
        )
    temporal_k, tuning = _autotune_selection_v1(
        tuning_path,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        event_transport=str(event_transport),
    )
    matrix = template.get("matrix")
    execution = template.get("execution")
    if not isinstance(matrix, Mapping) or not isinstance(execution, Mapping):
        raise CampaignProviderError("template matrix/execution is missing")
    direct_sparse = event_transport == DIRECT_SPARSE_EVENT_TRANSPORT_V3
    normalized_sparse = tuning.get("_provider_selection")
    requested = requested_batch_size
    if direct_sparse:
        if not isinstance(normalized_sparse, Mapping):
            raise CampaignProviderError(
                "direct-sparse autotune normalization is absent"
            )
        if purpose == "qualification" and (
            normalized_sparse.get("benchmark_status") != "FULL_TARGET_PASS"
            or normalized_sparse.get("minimum_committed_world_ticks_per_second", 0)
            < DIRECT_SPARSE_MINIMUM_COMMITTED_WORLD_TICKS_PER_SECOND_V3
        ):
            raise CampaignProviderError(
                "direct-sparse qualification requires the frozen minimum "
                "committed world-ticks/s target"
            )
        selected_from_tuning = int(normalized_sparse["batch_size"])
        if requested is not None and requested != selected_from_tuning:
            raise CampaignProviderError(
                "requested direct-sparse B differs from the selected autotune B"
            )
        requested = selected_from_tuning
    elif requested is None:
        requested = matrix.get("max_cells")
    if type(requested) is not int or requested <= 0:
        raise CampaignProviderError("requested batch ceiling must be positive")
    total_ticks = execution.get("total_ticks")
    start_tick = execution.get("start_next_tick", 0)
    if type(total_ticks) is not int or total_ticks <= 0 or type(start_tick) is not int:
        raise CampaignProviderError("template absolute tick horizon is invalid")
    requested_contract = CudaFullEvolutionNativeContractV2(
        B=requested,
        total_ticks=total_ticks,
        start_next_tick=start_tick,
        temporal_k=temporal_k,
    )
    admission = (
        None
        if direct_sparse
        else _cuda_v2_layout.select_cuda_native_batch_v2(
            requested_contract,
            available_vram_bytes=total_vram,
            reserve_bytes=reserve_bytes,
            include_reference_scratch=False,
        )
    )
    selected_batch = requested if direct_sparse else admission.selected_batch_size
    if selected_batch <= 0:
        raise CampaignProviderError("device cannot admit one CUDA Native V2 world")
    campaign_world_count = (
        logical_world_count or SHARED_W32_DEFAULT_LOGICAL_WORLDS_V1
        if shared_w32
        else selected_batch
    )
    if shared_w32 and not (
        SHARED_W32_MINIMUM_PHYSICAL_B_V1
        <= selected_batch
        <= SHARED_W32_MAXIMUM_PHYSICAL_B_V1
    ):
        raise CampaignProviderError("shared W32 requires physical B32-B136")
    if shared_w32 and temporal_k not in SHARED_W32_ALLOWED_TEMPORAL_K_V1:
        raise CampaignProviderError(
            "shared W32 temporal K must be one of K1, K2, K3, or K4"
        )
    if shared_w32 and campaign_world_count % selected_batch != 0:
        raise CampaignProviderError(
            "shared W32 requires uniform physical batches with no tail"
        )
    selected_contract = CudaFullEvolutionNativeContractV2(
        B=selected_batch,
        total_ticks=total_ticks,
        start_next_tick=start_tick,
        temporal_k=temporal_k,
    )
    if direct_sparse:
        usable_bytes = max(0, total_vram - reserve_bytes)
        maximum_working_bytes = (usable_bytes * 3) // 4
        required_bytes = int(normalized_sparse["estimated_peak_bytes"])
        if required_bytes > maximum_working_bytes:
            raise CampaignProviderError(
                "selected direct-sparse B/K exceeds fixed-total-VRAM admission"
            )
        headroom_bytes = usable_bytes - required_bytes
        initial_capacity_rows = int(
            normalized_sparse["initial_capacity_rows_per_world"]
        )
    else:
        selected_admission = _cuda_v2_layout.select_cuda_native_batch_v2(
            selected_contract,
            available_vram_bytes=total_vram,
            reserve_bytes=reserve_bytes,
            include_reference_scratch=False,
        )
        if not selected_admission.admitted:
            raise CampaignProviderError("selected CUDA Native V2 B is not admitted")
        maximum_working_bytes = selected_admission.maximum_working_bytes
        required_bytes = selected_admission.required_bytes
        headroom_bytes = selected_admission.headroom_bytes
        initial_capacity_rows = None
    output_spec = Path(output_spec_path).expanduser().resolve()
    output_request = Path(output_request_path).expanduser().resolve()
    selection_path = output_spec.with_suffix(".device-selection.json")
    if any(
        path.exists() or path.is_symlink()
        for path in (output_spec, output_request, selection_path)
    ):
        raise CampaignProviderError("concretizer outputs must be fresh paths")
    timestamp = created_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    selection_receipt = _selection_receipt_v1(
        artifact_sha256=artifact_sha256,
        manifest_file_sha256=manifest_file_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        tuning_path=tuning_path,
        tuning_file_sha256=_sha256_file(tuning_path),
        tuning_schema=str(tuning["schema_version"]),
        temporal_k=temporal_k,
        execution_mode=execution_mode,
        requested_batch_size=requested,
        selected_batch_size=selected_batch,
        reserve_bytes=reserve_bytes,
        required_bytes=required_bytes,
        maximum_working_bytes=maximum_working_bytes,
        headroom_bytes=headroom_bytes,
        created_at_utc=timestamp,
        event_transport=str(event_transport),
        initial_capacity_rows_per_world=initial_capacity_rows,
        autotune_result_sha256=(
            str(normalized_sparse["autotune_result_sha256"]) if direct_sparse else None
        ),
        selected_benchmark_body_sha256=(
            str(normalized_sparse["result_body_sha256"]) if direct_sparse else None
        ),
        selected_benchmark_file_sha256=(
            str(normalized_sparse["result_file_sha256"]) if direct_sparse else None
        ),
    )
    _write_bytes_durable(selection_path, canonical_bytes(selection_receipt))
    _seal_read_only_file_v1(selection_path)
    dimensions = matrix.get("dimensions")
    if not isinstance(dimensions, list):
        raise CampaignProviderError("template matrix dimensions are invalid")
    task_dimension = next(
        (
            row
            for row in dimensions
            if isinstance(row, Mapping) and row.get("id") == "task_seed"
        ),
        None,
    )
    if task_dimension is None:
        raise CampaignProviderError("template lacks the task_seed dimension")
    task_dimension["values"] = [
        100_001 + index for index in range(campaign_world_count)
    ]
    matrix["expected_cells"] = campaign_world_count
    matrix["max_cells"] = campaign_world_count
    parameters["cuda_native.temporal_k"] = temporal_k
    parameters["cuda_native.execution_mode"] = execution_mode
    parameters["cuda_native.event_transport"] = event_transport
    parameters["cuda_native.science_observation_mode"] = science_observation_mode
    execution["resident_owner_span_ticks"] = temporal_k
    execution["native_dispatch_window_ticks"] = temporal_k
    execution["source_window_ticks"] = temporal_k
    execution["event_window_capacity"] = (
        int(initial_capacity_rows) if direct_sparse else 1024 * temporal_k
    )
    execution["batch"] = {"mode": "explicit", "cells": selected_batch}
    if shared_w32:
        execution["gpu_resource_policy"] = SHARED_W32_GPU_POLICY_ID_V1
    output = template.get("output")
    if not isinstance(output, Mapping):
        raise CampaignProviderError("template output contract is invalid")
    output["runtime_root"] = str(Path(output_root).expanduser().resolve())
    checkpoint_transport = template.get("checkpoints")
    if not isinstance(checkpoint_transport, Mapping):
        raise CampaignProviderError("template checkpoint transport is invalid")
    checkpoint_transport["acs_slots"] = acs_checkpoint_slots
    checkpoint_transport["acs_capture_instrumentation"] = bool(
        acs_checkpoint_slots
    )
    checkpoint_transport["native_streamed_checkpoint"] = True
    ledger = (
        None
        if direct_sparse
        else _cuda_v2_layout.estimate_cuda_native_resources_v2(
            selected_contract, include_reference_scratch=False
        )
    )
    budgets = execution.get("budgets")
    if not isinstance(budgets, Mapping):
        raise CampaignProviderError("template execution budgets are invalid")
    # ``required_bytes`` is the sealed native owner allocation, not the whole
    # route budget once an optional acquisition sidecar is enabled.  Preserve
    # any larger author-declared envelope here; the exact direct-sparse
    # acquisition bound is folded in below once the observation mode and
    # snapshot schedule are known.
    budgets["vram_bytes"] = max(
        int(budgets.get("vram_bytes", 0)),
        required_bytes if direct_sparse else ledger.total_bytes,
    )
    checkpoints = execution.get("checkpoint_ticks")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or checkpoints[-1] != start_tick + total_ticks
    ):
        raise CampaignProviderError(
            "template must end with an absolute terminal checkpoint"
        )
    checkpoint_bytes = (
        campaign_world_count
        * _cuda_v2_layout.state_layout_v2().canonical_checkpoint_bytes_per_world
        + 128 * 1024
    ) * len(checkpoints)
    dispatches = (total_ticks + temporal_k - 1) // temporal_k
    event_bytes = (
        campaign_world_count
        * int(initial_capacity_rows)
        * _cuda_v2_layout.CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
        * dispatches
        if direct_sparse
        else selected_batch * 1024 * 48 * total_ticks
    )
    disk_bytes = checkpoint_bytes + event_bytes + 128 * 1024**2
    budgets["disk_bytes"] = max(int(budgets.get("disk_bytes", 0)), disk_bytes)
    lifecycle = template.get("artifact_lifecycle")
    if purpose == "qualification":
        existing_total = (
            lifecycle.get("storage", {}).get("total_max_bytes", 0)
            if isinstance(lifecycle, Mapping)
            else 0
        )
        scratch_required = event_bytes + 64 * 1024**2
        durable_required = 128 * 1024**2
        qualification_total = max(
            4,
            int(existing_total),
            disk_bytes,
            (scratch_required * 100 + 41) // 42,
            (checkpoint_bytes * 100 + 27) // 28,
            (durable_required * 100 + 22) // 23,
        )
        template["artifact_lifecycle"] = lifecycle_profile_contract_v1(
            QUALIFICATION,
            total_max_bytes=qualification_total,
            checkpoint_schedule=[int(value) for value in checkpoints],
            checkpoint_authority="absolute_tick",
        )
        lifecycle = template["artifact_lifecycle"]
    storage = lifecycle.get("storage") if isinstance(lifecycle, Mapping) else None
    if not isinstance(storage, Mapping):
        raise CampaignProviderError("template artifact lifecycle storage is invalid")
    if purpose == "qualification":
        if (
            int(storage.get("scratch_max_bytes", 0)) < scratch_required
            or int(storage.get("checkpoint_max_bytes", 0)) < checkpoint_bytes
            or int(storage.get("durable_max_bytes", 0)) < durable_required
        ):
            raise CampaignProviderError(
                "QUALIFICATION lifecycle partition cannot hold the sealed route"
            )
        template["artifact_lifecycle"] = validate_run_artifact_lifecycle_contract_v1(
            lifecycle
        )
    else:
        planning_estimate = (
            _direct_sparse_resource_estimate_v3(
                selected_contract,
                checkpoint_boundaries=len(checkpoints),
                sealed_selection=selection_receipt["selection"],
                device_selection=selection_receipt,
                acs_checkpoint_slots=acs_checkpoint_slots,
                compact_science_capture=(
                    science_observation_mode
                    == SCIENCE_OBSERVATION_COMPACT_SIDECAR_V1
                ),
                compact_snapshot_boundaries=len(
                    _science_snapshot_ticks_v1(template)
                ),
            )
            if direct_sparse
            else None
        )
        if planning_estimate is not None and shared_w32:
            planning_estimate = _aggregate_shared_w32_resource_estimate_v1(
                planning_estimate,
                physical_batch_count=campaign_world_count // selected_batch,
                logical_world_count=campaign_world_count,
            )
        if planning_estimate is not None:
            budgets["vram_bytes"] = max(
                int(budgets["vram_bytes"]),
                int(planning_estimate["vram_bytes_upper_bound"]),
            )
            budgets["wall_seconds"] = max(
                int(budgets["wall_seconds"]),
                int(planning_estimate["wall_seconds_upper_bound"]),
            )
        storage["scratch_max_bytes"] = event_bytes + 64 * 1024**2
        storage["checkpoint_max_bytes"] = (
            int(planning_estimate["checkpoint_bytes_upper_bound"])
            if planning_estimate is not None
            else checkpoint_bytes
        )
        storage["total_max_bytes"] = max(
            int(storage.get("total_max_bytes", 0)),
            (
                int(planning_estimate["disk_bytes_upper_bound"])
                + int(storage.get("emergency_reserve_bytes", 0))
                if planning_estimate is not None
                else disk_bytes
            ),
        )
        budgets["disk_bytes"] = max(
            int(budgets["disk_bytes"]),
            int(storage["total_max_bytes"]),
        )
        committed_without_durable = (
            int(storage["scratch_max_bytes"])
            + int(storage["checkpoint_max_bytes"])
            + int(storage.get("emergency_reserve_bytes", 0))
        )
        available_durable = int(storage["total_max_bytes"]) - committed_without_durable
        if available_durable < 1:
            raise CampaignProviderError(
                "non-qualification lifecycle cannot preserve a positive durable partition"
            )
        # This is a ceiling partition, not a reservation.  Assign all remaining
        # bytes to durable outputs so a later explicit-world lowering can add
        # bundle-declared compact snapshots without resealing device identity.
        storage["durable_max_bytes"] = available_durable
        template["artifact_lifecycle"] = validate_run_artifact_lifecycle_contract_v1(
            lifecycle
        )
    template["artifacts"] = {
        "cuda_extension": {
            "path": str(artifact),
            "sha256": artifact_sha256,
            "kind": "cuda_extension",
            "abi": CUDA_NATIVE_ABI_V1,
        },
        "extension_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_file_sha256,
            "kind": "extension_manifest",
            "abi": CUDA_NATIVE_ABI_V1,
        },
        "device_selection_receipt": {
            "path": str(selection_path),
            "sha256": _sha256_file(selection_path),
            "kind": "device_selection_receipt",
            "abi": CUDA_NATIVE_ABI_V1,
        },
    }
    if acs_native_extension is not None and acs_native_manifest is not None:
        template["artifacts"][ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1] = {
            "path": str(acs_native_extension),
            "sha256": _sha256_file(acs_native_extension),
            "kind": ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
            "abi": ACS_NATIVE_RUNTIME_ABI_V1,
        }
        template["artifacts"][ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1] = {
            "path": str(acs_native_manifest),
            "sha256": _sha256_file(acs_native_manifest),
            "kind": ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
            "abi": ACS_NATIVE_RUNTIME_ABI_V1,
        }
    if purpose == "qualification":
        template["description"] = (
            "Targeted delta Qualification execution candidate for the "
            "seven-capability CUDA Native V2 transition."
        )
        template["no_claims"] = (
            "TECHNICAL_ONLY_QUALIFICATION_EXECUTION. Executing this candidate "
            "does not itself grant Qualification, production, or science authority."
        )
    _reject_generation_authority_v1(template, path="campaign_spec")
    output_spec.parent.mkdir(parents=True, exist_ok=True)
    spec_bytes = yaml.safe_dump(template, sort_keys=False, allow_unicode=False).encode(
        "ascii"
    )
    _write_bytes_durable(output_spec, spec_bytes)
    _seal_read_only_file_v1(output_spec)
    baseline = {
        "schema_version": "eiel_baseline_declaration_v1",
        "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "intent": "qualification" if purpose == "qualification" else "development",
        "deviations": [
            {
                "requirement_id": "evolution.full_evolution",
                "reason": "CUDA Native V2 preserves all seven capabilities under an explicitly unqualified native route.",
            },
            {
                "requirement_id": "runtime.production_cuda",
                "reason": (
                    "The Qualification candidate provider is not yet the qualified "
                    "production provider."
                    if purpose == "qualification"
                    else "The native science/engineering provider is not the qualified production provider."
                ),
            },
            {
                "requirement_id": "science.wrc_variability",
                "reason": "The micro route has no admitted WRC science envelope.",
            },
            *(
                []
                if acs_checkpoint_slots == 2
                else [
                    {
                        "requirement_id": "checkpoint.acs_two_slot",
                        "reason": (
                            "CUDA Native V2 uses one detached ACS slot for this "
                            "execution; it is not the two-slot Baseline."
                            if acs_checkpoint_slots == 1
                            else (
                                "The native three-file absolute-tick checkpoint uses ACS0."
                            )
                        ),
                    }
                ]
            ),
            {
                "requirement_id": "authority.device_fallback",
                "reason": (
                    "The sealed device receipt replaces fallback authority for this "
                    f"{'Qualification candidate' if purpose == 'qualification' else 'non-qualification execution'} route."
                ),
            },
        ],
    }
    from .execution_front_door_v1 import ExecutionRequestV2

    request = ExecutionRequestV2.from_mapping(
        {
            "schema_version": "eiel_execution_request_v2",
            "purpose": purpose,
            "campaign_spec": str(output_spec),
            "output_root": str(Path(output_root).expanduser().resolve()),
            "resource_wait": {
                "enabled": False,
                "poll_seconds": 300,
                "max_wait_seconds": None,
            },
            "qualification_profile": None,
            "baseline": baseline,
        },
        base_dir=output_request.parent,
    )
    request_payload = request.payload()
    _reject_generation_authority_v1(request_payload, path="execution_request")
    _write_bytes_durable(output_request, canonical_bytes(request_payload))
    _seal_read_only_file_v1(output_request)
    return {
        "schema_version": "eiel_cuda_full_evolution_v2_concretization_v1",
        "campaign_spec_path": str(output_spec),
        "campaign_spec_sha256": _sha256_file(output_spec),
        "execution_request_path": str(output_request),
        "execution_request_sha256": request.request_sha256,
        "device_selection_receipt_path": str(selection_path),
        "device_selection_receipt_sha256": selection_receipt["receipt_sha256"],
        "artifact_sha256": artifact_sha256,
        "acs_native_runtime_extension_sha256": (
            _sha256_file(acs_native_extension)
            if acs_native_extension is not None
            else None
        ),
        "manifest_sha256": manifest_sha256,
        "device_identity": device_identity,
        "B": selected_batch,
        "temporal_k": temporal_k,
        "execution_mode": execution_mode,
        "event_transport": event_transport,
        "acs_checkpoint_slots": acs_checkpoint_slots,
        "gpu_resource_policy_id": (
            SHARED_W32_GPU_POLICY_ID_V1 if shared_w32 else None
        ),
        "logical_world_count": campaign_world_count,
        "physical_batch_count": campaign_world_count // selected_batch,
        "initial_capacity_rows_per_world": initial_capacity_rows,
        "horizon_dependent_resident_bytes": 0 if direct_sparse else None,
        "capacity_source": "exact_total_vram_fixed_reserve_and_25_percent_headroom",
        "transient_free_vram_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concretize one CUDA Native V2 candidate campaign/request"
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--acs-native-runtime-extension")
    parser.add_argument("--tuning-receipt", required=True)
    parser.add_argument("--spec-output", required=True)
    parser.add_argument("--request-output", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--requested-batch-size", type=int)
    parser.add_argument(
        "--reserve-bytes", type=int, default=DEFAULT_FIXED_VRAM_RESERVE_BYTES_V1
    )
    parser.add_argument("--purpose", default="qualprep")
    parser.add_argument("--event-transport", choices=EVENT_TRANSPORTS_V2)
    parser.add_argument("--acs-slots", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--gpu-resource-policy")
    parser.add_argument("--logical-worlds", type=int)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    receipt = concretize_device_spec_and_request_v1(
        arguments.template,
        artifact_path=arguments.artifact,
        acs_native_runtime_extension_path=(
            arguments.acs_native_runtime_extension
        ),
        tuning_receipt_path=arguments.tuning_receipt,
        output_spec_path=arguments.spec_output,
        output_request_path=arguments.request_output,
        output_root=arguments.output_root,
        requested_batch_size=arguments.requested_batch_size,
        reserve_bytes=arguments.reserve_bytes,
        purpose=arguments.purpose,
        event_transport=arguments.event_transport,
        acs_checkpoint_slots=arguments.acs_slots,
        gpu_resource_policy=arguments.gpu_resource_policy,
        logical_world_count=arguments.logical_worlds,
        device=arguments.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_AUTHORITY_NAME",
    "HOST_EXECUTION_COMPONENT_IDENTITY_SCHEMA_V1",
    "HOST_EXECUTION_COMPONENT_PATHS_V1",
    "PROFILE_ID",
    "PROVIDER_ID",
    "WORLD_BASE",
    "AbsoluteTickCheckpointOccurrenceAdapterV1",
    "CudaFullEvolutionV2ExecutionProviderV1",
    "concretize_device_spec_and_request_v1",
    "cuda_full_evolution_v2_profile_registry_v1",
    "cuda_full_evolution_v2_profile_v1",
    "cuda_full_evolution_v2_provider_registry_v1",
    "cuda_full_evolution_v2_materialized_resource_estimate_v1",
    "cuda_full_evolution_v2_signed_planning_resource_accounting_v1",
    "host_execution_component_identity_v1",
    "is_cuda_full_evolution_v2_spec_v1",
]
