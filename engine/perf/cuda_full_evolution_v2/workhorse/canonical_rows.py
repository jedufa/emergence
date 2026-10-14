"""Canonical-parent Workhorse V2 calibration, row, and aggregation contracts.

The outer campaign/Launchpad owner creates and validates Scheduler authority.
This module only reopens the caller-supplied sealed parent pointer and binds it
to exact Workhorse artifacts and rows.  It never imports Scheduler,
composition, provider, or CLI modules and never invokes an authority callback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Final

from .authority import (
    WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3,
    WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
    validate_workhorse_event_sink_receipt_v2,
)
from .contract import (
    FULL_EVOLUTION_CAPABILITIES_V2,
    WORKHORSE_CHECKPOINT_SCHEMA_V2,
    WORKHORSE_TIME_AUTHORITY_V2,
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
    workhorse_artifact_identity_v2,
    workhorse_backend_v2,
    workhorse_backend_temporal_k_candidates_v2,
    workhorse_shape_v2,
    workhorse_world_identity_binding_v2,
    validate_workhorse_world_identity_binding_v2,
)
from .layout import (
    WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2,
    WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2,
    WorkhorseOwnerComponentsV2,
    admit_workhorse_vram_v2,
    estimate_workhorse_resources_v2,
    workhorse_allocator_owner_overhead_bound_v2,
    workhorse_candidate_envelope_v2,
    workhorse_topology_resource_binding_v2,
    validate_workhorse_topology_resource_binding_v2,
)
from .native_loader import file_sha256, validate_workhorse_prebuilt_v2


WORKHORSE_CANONICAL_PARENT_BINDING_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_canonical_parent_attempt_binding_v2"
)
WORKHORSE_CANONICAL_TERMINAL_BINDING_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_canonical_parent_terminal_binding_v2"
)
WORKHORSE_CANONICAL_PREFLIGHT_REJECTION_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_canonical_preflight_rejection_binding_v2"
)
WORKHORSE_DEVICE_ENVELOPE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_device_envelope_calibration_receipt_v2"
)
WORKHORSE_CANONICAL_ROW_EXECUTION_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_canonical_row_execution_result_v3"
)
WORKHORSE_TECHNICAL_ROW_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_technical_row_receipt_v3"
)
WORKHORSE_FIXED_TICK_MEASUREMENT_SCHEDULE_SCHEMA_V3: Final[str] = (
    "eiel_workhorse_fixed_tick_measurement_schedule_v3"
)
WORKHORSE_TERMINALIZED_DEVICE_ENVELOPE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_terminalized_device_envelope_calibration_receipt_v2"
)
WORKHORSE_ROW_AGGREGATE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_offline_row_aggregate_v3"
)

_PARENT_SOURCE: Final[str] = "CANONICAL_PROVIDER_AFTER_SCHEDULER_REOPEN"
_CLAIM_CLASS: Final[str] = "TECHNICAL_ONLY"
_ROW_RATE_RELATIVE_TOLERANCE_V2: Final[float] = 1.0e-12
_ROW_RATE_ABSOLUTE_TOLERANCE_V2: Final[float] = 1.0e-9
_NO_AUTHORITY = {
    "execution_authority_granted": False,
    "qualification_authority_granted": False,
    "science_authority_granted": False,
}
_PARENT_COMMON_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authority_source",
        "execution_authority_granted",
        "exclusive_device",
        "attempt_authority_path",
        "attempt_authority_file_sha256",
        "scheduler_bundle_path",
        "scheduler_bundle_file_sha256",
        "scheduler_bundle_digest",
        "scheduler_live_heartbeat_sha256",
        "scheduler_live_heartbeat_at_utc",
        "run_id",
        "attempt_id",
        "source_batch_id",
        "attempt_lineage_sha256",
        "execution_request_sha256",
        "campaign_plan_sha256",
        "semantic_run_spec_sha256",
        "execution_implementation_sha256",
        "concretized_execution_sha256",
        "attempt_admission_request_sha256",
        "profile_id",
        "route_id",
        "shape_id",
        "backend_id",
        "artifact_sha256",
        "sealed_artifact_identity_sha256",
        "device_uuid",
        "device_identity",
        "owner_pid",
        "owner_process_start_ticks",
        "operation",
        "operation_contract_sha256",
        "contract",
        "compiled_temporal_k",
        "reserve_policy",
        "absolute_tick_span",
        "science_continuation_contract_sha256",
        "source_checkpoint_identity_sha256",
        "binding_sha256",
    }
)


class WorkhorseCanonicalRowError(ValueError):
    """A canonical-parent Workhorse receipt or row failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WorkhorseCanonicalRowError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0 or value > (1 << 63) - 1:
        raise WorkhorseCanonicalRowError(
            f"{label} must be a positive signed-int64 integer"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 63) - 1:
        raise WorkhorseCanonicalRowError(
            f"{label} must be a nonnegative signed-int64 integer"
        )
    return value


def workhorse_fixed_tick_measurement_schedule_v3(
    *,
    temporal_k: int,
    warmup_ticks: int,
    measurement_window_ticks: int,
    measured_windows: int,
) -> dict[str, Any]:
    """Create one K-comparable absolute-tick timing schedule."""

    k = _positive_int(temporal_k, "measurement temporal_k")
    warmup = _positive_int(warmup_ticks, "measurement warmup_ticks")
    window = _positive_int(
        measurement_window_ticks, "measurement measurement_window_ticks"
    )
    windows = _positive_int(measured_windows, "measurement measured_windows")
    if windows < 5:
        raise WorkhorseCanonicalRowError(
            "measurement measured_windows must be at least five"
        )
    if warmup % k or window % k:
        raise WorkhorseCanonicalRowError(
            "measurement warmup and window ticks must be divisible by temporal_k"
        )
    warmup_dispatches = warmup // k
    dispatches_per_window = window // k
    measured_dispatches = dispatches_per_window * windows
    measured_ticks = window * windows
    total_ticks = warmup + measured_ticks
    for label, item in (
        ("measurement measured_dispatches", measured_dispatches),
        ("measurement measured_ticks", measured_ticks),
        ("measurement total_ticks", total_ticks),
    ):
        _positive_int(item, label)
    body = {
        "schema_version": WORKHORSE_FIXED_TICK_MEASUREMENT_SCHEDULE_SCHEMA_V3,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "temporal_k": k,
        "warmup_ticks": warmup,
        "warmup_dispatches": warmup_dispatches,
        "measurement_window_ticks": window,
        "dispatches_per_measurement_window": dispatches_per_window,
        "measured_windows": windows,
        "measured_dispatches": measured_dispatches,
        "measured_ticks": measured_ticks,
        "total_ticks": total_ticks,
        "percentile_method": "HYNDMAN_FAN_R7_LINEAR_INTERPOLATION",
        "selection_statistic": (
            "p10_fixed_absolute_tick_window_world_ticks_per_second"
        ),
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def validate_workhorse_fixed_tick_measurement_schedule_v3(
    value: object, *, temporal_k: int
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "time_authority",
        "temporal_k",
        "warmup_ticks",
        "warmup_dispatches",
        "measurement_window_ticks",
        "dispatches_per_measurement_window",
        "measured_windows",
        "measured_dispatches",
        "measured_ticks",
        "total_ticks",
        "percentile_method",
        "selection_statistic",
        "schedule_sha256",
    }
    schedule = _exact_mapping(value, fields, "fixed-tick measurement schedule")
    expected = workhorse_fixed_tick_measurement_schedule_v3(
        temporal_k=temporal_k,
        warmup_ticks=schedule["warmup_ticks"],
        measurement_window_ticks=schedule["measurement_window_ticks"],
        measured_windows=schedule["measured_windows"],
    )
    if schedule != expected:
        raise WorkhorseCanonicalRowError("fixed-tick measurement schedule changed")
    return schedule


def workhorse_fixed_tick_comparison_identity_v3(
    *,
    schedule: Mapping[str, Any],
    start_next_tick: int,
    source_checkpoint_identity_sha256: str | None,
) -> dict[str, Any]:
    temporal_k = _positive_int(schedule.get("temporal_k"), "comparison temporal_k")
    validated = validate_workhorse_fixed_tick_measurement_schedule_v3(
        dict(schedule), temporal_k=temporal_k
    )
    start = _nonnegative_int(start_next_tick, "comparison start_next_tick")
    if source_checkpoint_identity_sha256 is not None:
        _sha(source_checkpoint_identity_sha256, "comparison source checkpoint")
    body = {
        "schema_version": "eiel_workhorse_fixed_tick_comparison_identity_v3",
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "start_next_tick": start,
        "warmup_ticks": validated["warmup_ticks"],
        "measurement_window_ticks": validated["measurement_window_ticks"],
        "measured_windows": validated["measured_windows"],
        "measured_ticks": validated["measured_ticks"],
        "total_ticks": validated["total_ticks"],
        "percentile_method": validated["percentile_method"],
        "selection_statistic": validated["selection_statistic"],
        "source_checkpoint_identity_sha256": source_checkpoint_identity_sha256,
    }
    return {**body, "comparison_identity_sha256": canonical_sha256(body)}


def _trimmed(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise WorkhorseCanonicalRowError(f"{label} must be a nonempty trimmed string")
    return value


def _exact_mapping(
    value: object, fields: set[str] | frozenset[str], label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise WorkhorseCanonicalRowError(f"{label} must contain exact fields")
    return dict(value)


def _reject_generation_authority(value: object, *, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise WorkhorseCanonicalRowError(f"{path} contains a non-string key")
            if "generation" in key.lower():
                raise WorkhorseCanonicalRowError(
                    f"{path}.{key} is generation-shaped; use absolute ticks"
                )
            _reject_generation_authority(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_generation_authority(child, path=f"{path}[{index}]")


def _regular_sealed_file(
    path_value: object,
    digest: object,
    label: str,
    *,
    require_read_only: bool = True,
) -> Path:
    path_text = _trimmed(path_value, f"{label} path")
    path = Path(path_text).expanduser()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WorkhorseCanonicalRowError(f"{label} is unavailable: {exc}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (
            require_read_only
            and metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )
    ):
        raise WorkhorseCanonicalRowError(
            f"{label} must be one read-only single-link regular file"
        )
    expected = _sha(digest, f"{label} file digest")
    if _sha256_file(path) != expected:
        raise WorkhorseCanonicalRowError(f"{label} file digest changed")
    return path.resolve()


def _write_sealed_json(value: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise WorkhorseCanonicalRowError("receipt output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("sealed receipt write made no progress")
            offset += written
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return output


def _read_sealed_receipt(
    receipt_path: str | Path, *, schema_version: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = Path(receipt_path).expanduser()
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseCanonicalRowError(f"cannot reopen {label}: {exc}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        or type(value) is not dict
    ):
        raise WorkhorseCanonicalRowError(
            f"{label} must be one sealed single-link JSON object"
        )
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    if value.get("schema_version") != schema_version or value.get(
        "receipt_content_sha256"
    ) != canonical_sha256(body):
        raise WorkhorseCanonicalRowError(f"{label} schema or content seal is invalid")
    value["receipt_path"] = str(path.resolve())
    value["receipt_file_sha256"] = _sha256_bytes(raw)
    return path.resolve(), value


def _validate_device_identity(value: object) -> dict[str, Any]:
    device = _exact_mapping(
        value,
        {
            "index",
            "name",
            "uuid",
            "pci_domain_id",
            "pci_bus_id",
            "pci_device_id",
            "total_memory_bytes",
            "multiprocessor_count",
            "sm",
            "cuda_runtime",
        },
        "device identity",
    )
    for field in ("index", "pci_domain_id", "pci_bus_id", "pci_device_id"):
        _nonnegative_int(device[field], f"device {field}")
    _positive_int(device["total_memory_bytes"], "device total_memory_bytes")
    _positive_int(device["multiprocessor_count"], "device multiprocessor_count")
    for field in ("name", "uuid", "sm", "cuda_runtime"):
        _trimmed(device[field], f"device {field}")
    return device


def workhorse_artifact_binding_v2(
    artifact_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Reopen one Workhorse prebuilt and return its exact row-facing identity."""
    artifact, manifest_path, manifest = validate_workhorse_prebuilt_v2(
        artifact_path,
        require_current_source_match=require_current_source_match,
    )
    shape = workhorse_shape_v2(str(manifest["shape_id"]))
    backend = workhorse_backend_v2(str(manifest["backend_id"]))
    identity = workhorse_artifact_identity_v2(shape, backend.backend_id)
    return {
        "path": str(artifact),
        "sha256": file_sha256(artifact),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "sealed_artifact_identity_sha256": str(
            manifest["sealed_artifact_identity"]["sealed_artifact_identity_sha256"]
        ),
        "source_inventory_sha256": str(manifest["source_inventory_sha256"]),
        "identity": identity,
    }


def _validate_artifact_binding(
    value: object, *, shape_id: str, backend_id: str
) -> dict[str, Any]:
    artifact = _exact_mapping(
        value,
        {
            "path",
            "sha256",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "sealed_artifact_identity_sha256",
            "source_inventory_sha256",
            "identity",
        },
        "artifact binding",
    )
    for field in (
        "sha256",
        "manifest_file_sha256",
        "manifest_sha256",
        "sealed_artifact_identity_sha256",
        "source_inventory_sha256",
    ):
        _sha(artifact[field], f"artifact {field}")
    _trimmed(artifact["path"], "artifact path")
    _trimmed(artifact["manifest_path"], "artifact manifest path")
    expected = workhorse_artifact_identity_v2(
        workhorse_shape_v2(shape_id), workhorse_backend_v2(backend_id).backend_id
    )
    if artifact["identity"] != expected:
        raise WorkhorseCanonicalRowError("artifact identity differs from shape/backend")
    return artifact


def _owner_components(
    value: object, *, backend_id: str, temporal_k: int
) -> WorkhorseOwnerComponentsV2:
    row = _exact_mapping(
        value,
        {
            "schema_version",
            "backend_id",
            "temporal_k",
            "graph_object_bytes",
            "graph_exec_bytes",
            "allocator_owner_overhead_bytes",
            "driver_unattributed_owner_bytes",
            "process_baseline_bytes",
            "checkpoint_device_staging_bytes_per_world",
            "measurement_receipt_sha256",
        },
        f"owner components K{temporal_k}",
    )
    components = WorkhorseOwnerComponentsV2(
        backend_id=str(row["backend_id"]),
        temporal_k=int(row["temporal_k"]),
        graph_object_bytes=int(row["graph_object_bytes"]),
        graph_exec_bytes=int(row["graph_exec_bytes"]),
        allocator_owner_overhead_bytes=int(row["allocator_owner_overhead_bytes"]),
        driver_unattributed_owner_bytes=int(row["driver_unattributed_owner_bytes"]),
        process_baseline_bytes=int(row["process_baseline_bytes"]),
        checkpoint_device_staging_bytes_per_world=int(
            row["checkpoint_device_staging_bytes_per_world"]
        ),
        measurement_receipt_sha256=str(row["measurement_receipt_sha256"]),
    )
    if (
        components.backend_id != backend_id
        or components.temporal_k != temporal_k
        or components.to_dict() != row
    ):
        raise WorkhorseCanonicalRowError("owner components changed identity")
    return components


def validate_workhorse_canonical_parent_authority_binding_v2(
    value: Mapping[str, Any],
    *,
    expected_operation: str | None = None,
) -> dict[str, Any]:
    """Validate the exact opaque pointer projection supplied by Launchpad.

    This validation proves Workhorse binding and exact pointer bytes only.  It
    is not an authority check: the canonical provider must construct the
    mapping after directly reopening and heartbeating the current Scheduler
    bundle, and must later reopen Scheduler terminalization.
    """
    binding = _exact_mapping(value, _PARENT_COMMON_FIELDS, "canonical parent binding")
    _reject_generation_authority(binding, path="canonical_parent_binding")
    if (
        binding["schema_version"] != WORKHORSE_CANONICAL_PARENT_BINDING_SCHEMA_V2
        or binding["authority_source"] != _PARENT_SOURCE
        # This object is a provider-internal execution context, not an
        # authority token.  The canonical provider has already reopened the
        # Scheduler pointer and performed the live heartbeat; callers cannot
        # gain execution authority by manufacturing this mapping.
        or binding["execution_authority_granted"] is not False
        or binding["exclusive_device"] is not True
    ):
        raise WorkhorseCanonicalRowError("canonical parent authority flags are invalid")
    for field in (
        "attempt_authority_file_sha256",
        "scheduler_bundle_file_sha256",
        "scheduler_bundle_digest",
        "scheduler_live_heartbeat_sha256",
        "attempt_lineage_sha256",
        "execution_request_sha256",
        "campaign_plan_sha256",
        "semantic_run_spec_sha256",
        "execution_implementation_sha256",
        "concretized_execution_sha256",
        "attempt_admission_request_sha256",
        "artifact_sha256",
        "sealed_artifact_identity_sha256",
        "operation_contract_sha256",
    ):
        _sha(binding[field], f"canonical parent {field}")
    for field in (
        "run_id",
        "attempt_id",
        "source_batch_id",
        "profile_id",
        "route_id",
        "shape_id",
        "backend_id",
        "device_uuid",
    ):
        _trimmed(binding[field], f"canonical parent {field}")
    _positive_int(binding["owner_pid"], "canonical parent owner_pid")
    _positive_int(
        binding["owner_process_start_ticks"],
        "canonical parent owner_process_start_ticks",
    )
    _regular_sealed_file(
        binding["attempt_authority_path"],
        binding["attempt_authority_file_sha256"],
        "canonical Attempt pointer",
    )
    _regular_sealed_file(
        binding["scheduler_bundle_path"],
        binding["scheduler_bundle_file_sha256"],
        "canonical Scheduler bundle",
        require_read_only=False,
    )
    _trimmed(
        binding["scheduler_live_heartbeat_at_utc"],
        "canonical Scheduler heartbeat time",
    )
    operation = binding["operation"]
    if operation not in {
        "DEVICE_ENVELOPE_CALIBRATION",
        "EXECUTION_ROW",
        "SEMANTIC_EVIDENCE_CASE",
        "SCIENCE_COMPATIBILITY_DEMO",
        "INDEXED_ACS_CONTINUATION",
    }:
        raise WorkhorseCanonicalRowError("canonical parent operation is unsupported")
    if expected_operation is not None and operation != expected_operation:
        raise WorkhorseCanonicalRowError("canonical parent operation differs")
    shape = workhorse_shape_v2(str(binding["shape_id"]))
    backend = workhorse_backend_v2(str(binding["backend_id"]))
    device = _validate_device_identity(binding["device_identity"])
    if device["uuid"] != binding["device_uuid"]:
        raise WorkhorseCanonicalRowError("canonical parent device identity changed")
    compiled_k = binding["compiled_temporal_k"]
    if (
        binding["profile_id"] != shape.profile_id
        or binding["route_id"] != shape.route_id
    ):
        raise WorkhorseCanonicalRowError("canonical parent shape/profile/K changed")
    if operation == "DEVICE_ENVELOPE_CALIBRATION":
        if (
            type(compiled_k) is not list
            or compiled_k
            != list(
                workhorse_backend_temporal_k_candidates_v2(shape, backend.backend_id)
            )
            or len(set(compiled_k)) != len(compiled_k)
            or binding["contract"] is not None
            or binding["science_continuation_contract_sha256"] is not None
            or binding["source_checkpoint_identity_sha256"] is not None
        ):
            raise WorkhorseCanonicalRowError("envelope calibration cannot bind one row")
        reserve_policy = _exact_mapping(
            binding["reserve_policy"],
            {
                "mode",
                "reserve_bytes_requested",
                "minimum_fraction_numerator",
                "minimum_fraction_denominator",
                "transient_free_bytes_is_plan_authority",
            },
            "calibration reserve policy",
        )
        if (
            reserve_policy["mode"] != "STABLE_TOTAL_MINUS_EFFECTIVE_RESERVE_V2"
            or reserve_policy["minimum_fraction_numerator"]
            != WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
            or reserve_policy["minimum_fraction_denominator"]
            != WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
            or reserve_policy["transient_free_bytes_is_plan_authority"] is not False
        ):
            raise WorkhorseCanonicalRowError("calibration reserve policy changed")
        _nonnegative_int(
            reserve_policy["reserve_bytes_requested"], "calibration reserve bytes"
        )
    else:
        if type(binding["contract"]) is not dict:
            raise WorkhorseCanonicalRowError(
                "execution or semantic parent lacks its contract"
            )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            binding["contract"]
        )
        continuation_compiled_k_valid = (
            operation == "INDEXED_ACS_CONTINUATION"
            and type(compiled_k) is list
            and compiled_k == sorted(set(compiled_k))
            and all(type(value) is int and value > 0 for value in compiled_k)
            and 1 in compiled_k
            and contract.temporal_k in compiled_k
        )
        fixed_compiled_k_valid = (
            operation != "INDEXED_ACS_CONTINUATION"
            and type(compiled_k) is int
            and compiled_k == contract.temporal_k
        )
        if (
            contract.shape_id != shape.shape_id
            or contract.backend_id != backend.backend_id
            or not (continuation_compiled_k_valid or fixed_compiled_k_valid)
        ):
            raise WorkhorseCanonicalRowError(
                "canonical parent contract changed execution operation"
            )
        if binding["reserve_policy"] is not None:
            raise WorkhorseCanonicalRowError(
                "execution operation gained reserve authority"
            )
        _sha(
            binding["science_continuation_contract_sha256"],
            "canonical parent science continuation contract",
        )
    span = _exact_mapping(
        binding["absolute_tick_span"],
        {
            "time_authority",
            "start_next_tick",
            "target_next_tick",
            "total_ticks",
        },
        "canonical parent absolute-tick span",
    )
    start = _nonnegative_int(span["start_next_tick"], "parent start tick")
    total = _positive_int(span["total_ticks"], "parent total ticks")
    target = _positive_int(span["target_next_tick"], "parent target tick")
    if (
        span["time_authority"] != WORKHORSE_TIME_AUTHORITY_V2
        or target != start + total
        or (
            operation
            in {
                "EXECUTION_ROW",
                "SEMANTIC_EVIDENCE_CASE",
                "SCIENCE_COMPATIBILITY_DEMO",
                "INDEXED_ACS_CONTINUATION",
            }
            and (start != contract.start_next_tick or total != contract.total_ticks)
        )
    ):
        raise WorkhorseCanonicalRowError("canonical parent absolute-tick span changed")
    source_checkpoint = binding["source_checkpoint_identity_sha256"]
    if source_checkpoint is not None:
        _sha(source_checkpoint, "canonical parent source checkpoint identity")
    supplied = _sha(binding["binding_sha256"], "canonical parent binding digest")
    core = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if supplied != canonical_sha256(core):
        raise WorkhorseCanonicalRowError("canonical parent binding digest mismatch")
    return binding


def validate_workhorse_live_device_under_semantic_parent_v2(
    canonical_parent_authority_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the exact ten-field live CUDA identity before one semantic case."""

    return validate_workhorse_live_device_under_parent_v2(
        canonical_parent_authority_binding,
        expected_operation="SEMANTIC_EVIDENCE_CASE",
    )


def validate_workhorse_live_device_under_parent_v2(
    canonical_parent_authority_binding: Mapping[str, Any],
    *,
    expected_operation: str,
) -> dict[str, Any]:
    """Reopen the exact live CUDA identity for one canonical operation."""

    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation=expected_operation,
    )
    from .probe import _torch
    from ..benchmark import cuda_device_identity_v2

    torch = _torch()
    device = _validate_device_identity(
        cuda_device_identity_v2(torch, torch.cuda.current_device())
    )
    if device != parent["device_identity"]:
        raise WorkhorseCanonicalRowError(
            "runtime device differs from its canonical parent"
        )
    return device


def _validate_parent_for_artifact(
    parent: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    identity = artifact["identity"]
    if (
        parent["shape_id"] != identity["shape_id"]
        or parent["backend_id"] != identity["backend_id"]
        or parent["profile_id"] != identity["profile_id"]
        or parent["route_id"] != identity["route_id"]
        or parent["artifact_sha256"] != artifact["sha256"]
        or parent["sealed_artifact_identity_sha256"]
        != artifact["sealed_artifact_identity_sha256"]
    ):
        raise WorkhorseCanonicalRowError(
            "canonical parent binds another Workhorse artifact"
        )


def _capacity_basis(total_device_bytes: int, reserve_bytes: int) -> dict[str, Any]:
    _positive_int(total_device_bytes, "total_device_bytes")
    _nonnegative_int(reserve_bytes, "reserve_bytes")
    fractional = (
        total_device_bytes * WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
        + WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
        - 1
    ) // WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
    effective = max(reserve_bytes, fractional)
    return {
        "mode": "STABLE_TOTAL_MINUS_EFFECTIVE_RESERVE_V2",
        "total_device_bytes": total_device_bytes,
        "reserve_bytes_requested": reserve_bytes,
        "minimum_fraction_numerator": WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2,
        "minimum_fraction_denominator": WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2,
        "effective_reserve_bytes": effective,
        "planning_available_bytes": max(0, total_device_bytes - effective),
        "transient_free_bytes_is_plan_authority": False,
    }


def _validate_owner_measurement(
    value: object,
    *,
    components: WorkhorseOwnerComponentsV2,
    parent: Mapping[str, Any],
    artifact: Mapping[str, Any],
    device: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = _exact_mapping(
        value,
        {
            "schema_version",
            "measurement_source",
            "accounting_mode",
            "canonical_parent_binding_sha256",
            "artifact_sha256",
            "sealed_artifact_identity_sha256",
            "device_uuid",
            "backend_id",
            "temporal_k",
            "free_before_bytes",
            "free_with_owner_bytes",
            "total_device_bytes",
            "tensor_resident_bytes",
            "allocator_reserved_baseline_bytes",
            "allocator_reserved_with_owner_bytes",
            "allocator_observed_reserved_overhead_bytes",
            "allocator_explicit_bound_floor_bytes",
            "allocator_owner_overhead_bytes",
            "graph_object_bytes",
            "graph_exec_bytes",
            "driver_unattributed_owner_bytes",
            "process_baseline_bytes",
            "graph_inspection",
            "owner_components",
            "measurement_content_sha256",
        },
        f"owner measurement K{components.temporal_k}",
    )
    body = {
        key: item
        for key, item in measurement.items()
        if key not in {"measurement_content_sha256", "owner_components"}
    }
    if (
        measurement["schema_version"]
        != "eiel_workhorse_owner_component_calibration_under_parent_v2"
        or measurement["measurement_source"]
        != "CANONICAL_EXCLUSIVE_PARENT_CUDA_GLOBAL_DELTA"
        or measurement["accounting_mode"] != "LEASE_BACKED_GLOBAL_DELTA"
        or measurement["canonical_parent_binding_sha256"] != parent["binding_sha256"]
        or measurement["artifact_sha256"] != artifact["sha256"]
        or measurement["sealed_artifact_identity_sha256"]
        != artifact["sealed_artifact_identity_sha256"]
        or measurement["device_uuid"] != device["uuid"]
        or measurement["backend_id"] != components.backend_id
        or measurement["temporal_k"] != components.temporal_k
        or measurement["total_device_bytes"] != device["total_memory_bytes"]
        or measurement["owner_components"] != components.to_dict()
        or measurement["measurement_content_sha256"] != canonical_sha256(body)
        or components.measurement_receipt_sha256
        != measurement["measurement_content_sha256"]
    ):
        raise WorkhorseCanonicalRowError("owner measurement identity/seal changed")
    for field in (
        "free_before_bytes",
        "free_with_owner_bytes",
        "total_device_bytes",
        "tensor_resident_bytes",
        "allocator_reserved_baseline_bytes",
        "allocator_reserved_with_owner_bytes",
        "allocator_observed_reserved_overhead_bytes",
        "allocator_explicit_bound_floor_bytes",
        "allocator_owner_overhead_bytes",
        "graph_object_bytes",
        "graph_exec_bytes",
        "driver_unattributed_owner_bytes",
        "process_baseline_bytes",
    ):
        _nonnegative_int(measurement[field], f"owner measurement {field}")
    if (
        measurement["free_before_bytes"] > measurement["total_device_bytes"]
        or measurement["free_with_owner_bytes"] > measurement["total_device_bytes"]
        or measurement["allocator_observed_reserved_overhead_bytes"]
        != max(
            0,
            measurement["allocator_reserved_with_owner_bytes"]
            - measurement["allocator_reserved_baseline_bytes"]
            - measurement["tensor_resident_bytes"],
        )
        or measurement["allocator_explicit_bound_floor_bytes"]
        != workhorse_allocator_owner_overhead_bound_v2(0)
        or measurement["allocator_owner_overhead_bytes"]
        != workhorse_allocator_owner_overhead_bound_v2(
            measurement["allocator_observed_reserved_overhead_bytes"]
        )
        or measurement["allocator_owner_overhead_bytes"]
        != components.allocator_owner_overhead_bytes
        or measurement["graph_object_bytes"] != components.graph_object_bytes
        or measurement["graph_exec_bytes"] != components.graph_exec_bytes
        or measurement["driver_unattributed_owner_bytes"]
        != components.driver_unattributed_owner_bytes
        or measurement["process_baseline_bytes"] != components.process_baseline_bytes
    ):
        raise WorkhorseCanonicalRowError("owner measurement arithmetic changed")
    return measurement


def run_and_seal_workhorse_device_envelope_calibration_v2(
    artifact_path: str | Path,
    *,
    output_path: str | Path,
    source_identity: Mapping[str, Any],
    canonical_parent_authority_binding: Mapping[str, Any],
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Measure every compiled K under one canonical calibration Attempt.

    This is physical-CUDA work and is intentionally not invoked by import,
    validation, or aggregation.  The parent Attempt's exclusive-device lease
    makes CUDA global free-memory deltas attributable to this calibration
    owner without a per-substep Scheduler callback.
    """
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation="DEVICE_ENVELOPE_CALIBRATION",
    )
    artifact = workhorse_artifact_binding_v2(artifact_path)
    _validate_parent_for_artifact(parent, artifact)

    from .probe import (
        _allocate_owner,
        _initialize_owner,
        _load,
        _prepare_graph_if_needed,
        _release_graph,
        _torch,
    )
    from ..benchmark import cuda_device_identity_v2

    torch = _torch()
    loaded = _load(artifact_path, torch)
    runtime = dict(loaded.module.runtime_inspect())
    device = cuda_device_identity_v2(torch, torch.cuda.current_device())
    device = _validate_device_identity(device)
    if device != parent["device_identity"]:
        raise WorkhorseCanonicalRowError("calibration device differs from parent")
    components_by_k: dict[int, WorkhorseOwnerComponentsV2] = {}
    measurements_by_k: dict[int, dict[str, Any]] = {}
    topology_by_k: dict[int, dict[str, Any]] = {}
    for temporal_k in workhorse_backend_temporal_k_candidates_v2(
        loaded.shape, loaded.backend.backend_id
    ):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        device_index = torch.cuda.current_device()
        free_before, total_before = torch.cuda.mem_get_info(device_index)
        if int(total_before) != device["total_memory_bytes"]:
            raise WorkhorseCanonicalRowError(
                "calibration total VRAM differs from sealed device"
            )
        reserved_before = int(torch.cuda.memory_reserved(device_index))
        owner = None
        graph_inspection = None
        try:
            owner = _allocate_owner(torch, loaded, 1, temporal_k)
            _initialize_owner(
                loaded,
                owner,
                world_uid_base=209_000,
                start_tick=0,
                rng_key=79,
                fixture=False,
            )
            graph_inspection = _prepare_graph_if_needed(loaded, owner, temporal_k, 0)
            torch.cuda.synchronize()
            free_with_owner, total_with_owner = torch.cuda.mem_get_info(device_index)
            if int(total_with_owner) != int(total_before):
                raise WorkhorseCanonicalRowError(
                    "calibration device total changed while owner was live"
                )
            tensor_bytes = (
                loaded.layout.state_bytes_per_world
                + loaded.layout.scratch_bytes_per_world
                + loaded.layout.event_bytes_per_world_tick * temporal_k
                + 4
            )
            reserved_with_owner = int(torch.cuda.memory_reserved(device_index))
            allocator_observed_reserved_overhead = max(
                0, reserved_with_owner - reserved_before - tensor_bytes
            )
            allocator_explicit_bound_floor = (
                workhorse_allocator_owner_overhead_bound_v2(0)
            )
            allocator_overhead = workhorse_allocator_owner_overhead_bound_v2(
                allocator_observed_reserved_overhead
            )
            graph_object_bytes = (
                0
                if graph_inspection is None
                else int(graph_inspection["graph_object_device_bytes"])
            )
            graph_exec_bytes = (
                0
                if graph_inspection is None
                else int(graph_inspection["graph_exec_device_bytes"])
            )
            global_owner_delta = max(0, int(free_before) - int(free_with_owner))
            accounted = (
                tensor_bytes
                + allocator_overhead
                + graph_object_bytes
                + graph_exec_bytes
            )
            driver_residual = max(0, global_owner_delta - accounted)
            process_baseline = max(0, int(total_before) - int(free_before))
            measurement_body = {
                "schema_version": (
                    "eiel_workhorse_owner_component_calibration_under_parent_v2"
                ),
                "measurement_source": ("CANONICAL_EXCLUSIVE_PARENT_CUDA_GLOBAL_DELTA"),
                "accounting_mode": "LEASE_BACKED_GLOBAL_DELTA",
                "canonical_parent_binding_sha256": parent["binding_sha256"],
                "artifact_sha256": artifact["sha256"],
                "sealed_artifact_identity_sha256": artifact[
                    "sealed_artifact_identity_sha256"
                ],
                "device_uuid": device["uuid"],
                "backend_id": loaded.backend.backend_id,
                "temporal_k": temporal_k,
                "free_before_bytes": int(free_before),
                "free_with_owner_bytes": int(free_with_owner),
                "total_device_bytes": int(total_before),
                "tensor_resident_bytes": tensor_bytes,
                "allocator_reserved_baseline_bytes": reserved_before,
                "allocator_reserved_with_owner_bytes": reserved_with_owner,
                "allocator_observed_reserved_overhead_bytes": (
                    allocator_observed_reserved_overhead
                ),
                "allocator_explicit_bound_floor_bytes": (
                    allocator_explicit_bound_floor
                ),
                "allocator_owner_overhead_bytes": allocator_overhead,
                "graph_object_bytes": graph_object_bytes,
                "graph_exec_bytes": graph_exec_bytes,
                "driver_unattributed_owner_bytes": driver_residual,
                "process_baseline_bytes": process_baseline,
                "graph_inspection": graph_inspection,
            }
            measurement_sha256 = canonical_sha256(measurement_body)
            components = WorkhorseOwnerComponentsV2(
                backend_id=loaded.backend.backend_id,
                temporal_k=temporal_k,
                graph_object_bytes=graph_object_bytes,
                graph_exec_bytes=graph_exec_bytes,
                allocator_owner_overhead_bytes=allocator_overhead,
                driver_unattributed_owner_bytes=driver_residual,
                process_baseline_bytes=process_baseline,
                checkpoint_device_staging_bytes_per_world=0,
                measurement_receipt_sha256=measurement_sha256,
            )
            measurement = {
                **measurement_body,
                "owner_components": components.to_dict(),
                "measurement_content_sha256": measurement_sha256,
            }
            components_by_k[temporal_k] = components
            measurements_by_k[temporal_k] = measurement
            topology_by_k[temporal_k] = workhorse_topology_resource_binding_v2(
                loaded.shape,
                backend_id=loaded.backend.backend_id,
                temporal_k=temporal_k,
                runtime_inspection=runtime,
                graph_inspection=graph_inspection,
            )
        finally:
            _release_graph(loaded)
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return seal_workhorse_device_envelope_calibration_receipt_v2(
        output_path,
        artifact_binding=artifact,
        source_identity=source_identity,
        device_identity=device,
        owner_components_by_k=components_by_k,
        owner_component_measurements_by_k=measurements_by_k,
        topology_resource_bindings_by_k=topology_by_k,
        canonical_parent_authority_binding=parent,
        reserve_bytes=reserve_bytes,
    )


def seal_workhorse_device_envelope_calibration_receipt_v2(
    output_path: str | Path,
    *,
    artifact_binding: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    owner_components_by_k: Mapping[int, WorkhorseOwnerComponentsV2],
    owner_component_measurements_by_k: Mapping[int, Mapping[str, Any]],
    topology_resource_bindings_by_k: Mapping[int, Mapping[str, Any]],
    canonical_parent_authority_binding: Mapping[str, Any],
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Seal one stable-total device envelope covering every compiled K."""
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation="DEVICE_ENVELOPE_CALIBRATION",
    )
    shape = workhorse_shape_v2(str(parent["shape_id"]))
    backend = workhorse_backend_v2(str(parent["backend_id"]))
    artifact = _validate_artifact_binding(
        artifact_binding, shape_id=shape.shape_id, backend_id=backend.backend_id
    )
    _validate_parent_for_artifact(parent, artifact)
    source = dict(source_identity)
    if not source:
        raise WorkhorseCanonicalRowError("source identity must be nonempty")
    _reject_generation_authority(source, path="source_identity")
    device = _validate_device_identity(device_identity)
    if device != parent["device_identity"]:
        raise WorkhorseCanonicalRowError("device identity differs from parent Attempt")
    expected_k = workhorse_backend_temporal_k_candidates_v2(shape, backend.backend_id)
    if set(owner_components_by_k) != set(expected_k):
        raise WorkhorseCanonicalRowError(
            "owner components must cover every compiled K exactly"
        )
    if set(owner_component_measurements_by_k) != set(expected_k):
        raise WorkhorseCanonicalRowError(
            "owner measurements must cover every compiled K exactly"
        )
    if set(topology_resource_bindings_by_k) != set(expected_k):
        raise WorkhorseCanonicalRowError(
            "topology bindings must cover every compiled K exactly"
        )
    components: dict[int, WorkhorseOwnerComponentsV2] = {}
    measurements: dict[int, dict[str, Any]] = {}
    topology: dict[int, dict[str, Any]] = {}
    for temporal_k in expected_k:
        component = owner_components_by_k[temporal_k]
        if type(component) is not WorkhorseOwnerComponentsV2:
            raise WorkhorseCanonicalRowError("owner component type is not exact")
        if (
            component.backend_id != backend.backend_id
            or component.temporal_k != temporal_k
        ):
            raise WorkhorseCanonicalRowError("owner component backend/K changed")
        components[temporal_k] = component
        measurements[temporal_k] = _validate_owner_measurement(
            owner_component_measurements_by_k[temporal_k],
            components=component,
            parent=parent,
            artifact=artifact,
            device=device,
        )
        topology[temporal_k] = validate_workhorse_topology_resource_binding_v2(
            topology_resource_bindings_by_k[temporal_k],
            shape=shape,
            backend_id=backend.backend_id,
            temporal_k=temporal_k,
        )
        limits = topology[temporal_k]["device_limits"]
        if limits["multiprocessor_count"] != device["multiprocessor_count"]:
            raise WorkhorseCanonicalRowError("topology differs from device identity")
    capacity = _capacity_basis(device["total_memory_bytes"], reserve_bytes)
    # Candidate generation receives total bytes as its planning-free basis.
    # Runtime free memory is deliberately checked again by each started row.
    envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=backend.backend_id,
        free_bytes=device["total_memory_bytes"],
        total_device_bytes=device["total_memory_bytes"],
        owner_components_by_k=components,
        topology_resource_bindings_by_k=topology,
        live_observation_authority_binding_sha256=parent["binding_sha256"],
        reserve_bytes=reserve_bytes,
    )
    expected_rows = [
        {
            "shape_id": shape.shape_id,
            "backend_id": backend.backend_id,
            "temporal_k": k_row["temporal_k"],
            "B": row["B"],
            "planned_admission": row["admission"],
        }
        for k_row in envelope["per_k"]
        for row in k_row["rows"]
    ]
    body = {
        "schema_version": WORKHORSE_DEVICE_ENVELOPE_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
        "shape_id": shape.shape_id,
        "backend_id": backend.backend_id,
        "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "artifact": artifact,
        "source_identity": source,
        "source_identity_sha256": canonical_sha256(source),
        "device_identity": device,
        "capacity_basis": capacity,
        "canonical_parent_authority_binding": parent,
        "owner_components_by_k": {
            str(key): value.to_dict() for key, value in components.items()
        },
        "owner_component_measurements_by_k": {
            str(key): value for key, value in measurements.items()
        },
        "topology_resource_bindings_by_k": {
            str(key): value for key, value in topology.items()
        },
        "candidate_envelope": envelope,
        "expected_rows": expected_rows,
        "expected_row_count": len(expected_rows),
        "checkpoint_and_science_identity": {
            "checkpoint_schema": WORKHORSE_CHECKPOINT_SCHEMA_V2,
            "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
            "absolute_tick_only": True,
            "source_checkpoint_identity_required_on_restore": True,
            "science_continuation_binding_required_for_future_science": True,
            "current_receipt_is_science_authority": False,
        },
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_device_envelope_calibration_receipt_v2(output)


def validate_workhorse_device_envelope_calibration_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen and recompute a complete stable-total calibration envelope."""
    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_DEVICE_ENVELOPE_SCHEMA_V2,
        label="Workhorse device-envelope receipt",
    )
    _reject_generation_authority(value, path="device_envelope")
    if (
        value.get("claim_class") != _CLAIM_CLASS
        or any(value.get(field) is not False for field in _NO_AUTHORITY)
        or value.get("capabilities") != list(FULL_EVOLUTION_CAPABILITIES_V2)
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
    ):
        raise WorkhorseCanonicalRowError("device-envelope authority flags changed")
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        value.get("canonical_parent_authority_binding"),
        expected_operation="DEVICE_ENVELOPE_CALIBRATION",
    )
    shape = workhorse_shape_v2(str(value.get("shape_id")))
    backend = workhorse_backend_v2(str(value.get("backend_id")))
    artifact = _validate_artifact_binding(
        value.get("artifact"), shape_id=shape.shape_id, backend_id=backend.backend_id
    )
    _validate_parent_for_artifact(parent, artifact)
    source = value.get("source_identity")
    if type(source) is not dict or not source:
        raise WorkhorseCanonicalRowError("device-envelope source identity is absent")
    _reject_generation_authority(source, path="source_identity")
    if value.get("source_identity_sha256") != canonical_sha256(source):
        raise WorkhorseCanonicalRowError("device-envelope source identity changed")
    device = _validate_device_identity(value.get("device_identity"))
    if device != parent["device_identity"]:
        raise WorkhorseCanonicalRowError("device-envelope parent device changed")
    capacity = _capacity_basis(
        device["total_memory_bytes"],
        _nonnegative_int(
            value.get("capacity_basis", {}).get("reserve_bytes_requested"),
            "capacity reserve",
        ),
    )
    if value.get("capacity_basis") != capacity:
        raise WorkhorseCanonicalRowError("stable capacity basis changed")
    expected_k = workhorse_backend_temporal_k_candidates_v2(shape, backend.backend_id)
    raw_components = value.get("owner_components_by_k")
    raw_measurements = value.get("owner_component_measurements_by_k")
    raw_topology = value.get("topology_resource_bindings_by_k")
    if (
        type(raw_components) is not dict
        or set(raw_components) != {str(item) for item in expected_k}
        or type(raw_measurements) is not dict
        or set(raw_measurements) != {str(item) for item in expected_k}
        or type(raw_topology) is not dict
        or set(raw_topology) != {str(item) for item in expected_k}
    ):
        raise WorkhorseCanonicalRowError("device-envelope K coverage is incomplete")
    components: dict[int, WorkhorseOwnerComponentsV2] = {}
    topology: dict[int, dict[str, Any]] = {}
    for temporal_k in expected_k:
        components[temporal_k] = _owner_components(
            raw_components[str(temporal_k)],
            backend_id=backend.backend_id,
            temporal_k=temporal_k,
        )
        _validate_owner_measurement(
            raw_measurements[str(temporal_k)],
            components=components[temporal_k],
            parent=parent,
            artifact=artifact,
            device=device,
        )
        topology[temporal_k] = validate_workhorse_topology_resource_binding_v2(
            raw_topology[str(temporal_k)],
            shape=shape,
            backend_id=backend.backend_id,
            temporal_k=temporal_k,
        )
        if (
            topology[temporal_k]["device_limits"]["multiprocessor_count"]
            != device["multiprocessor_count"]
        ):
            raise WorkhorseCanonicalRowError("device-envelope topology changed device")
    envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=backend.backend_id,
        free_bytes=device["total_memory_bytes"],
        total_device_bytes=device["total_memory_bytes"],
        owner_components_by_k=components,
        topology_resource_bindings_by_k=topology,
        live_observation_authority_binding_sha256=parent["binding_sha256"],
        reserve_bytes=capacity["reserve_bytes_requested"],
    )
    expected_rows = [
        {
            "shape_id": shape.shape_id,
            "backend_id": backend.backend_id,
            "temporal_k": k_row["temporal_k"],
            "B": row["B"],
            "planned_admission": row["admission"],
        }
        for k_row in envelope["per_k"]
        for row in k_row["rows"]
    ]
    if (
        value.get("profile_id") != shape.profile_id
        or value.get("route_id") != shape.route_id
        or value.get("candidate_envelope") != envelope
        or value.get("expected_rows") != expected_rows
        or value.get("expected_row_count") != len(expected_rows)
        or value.get("checkpoint_and_science_identity")
        != {
            "checkpoint_schema": WORKHORSE_CHECKPOINT_SCHEMA_V2,
            "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
            "absolute_tick_only": True,
            "source_checkpoint_identity_required_on_restore": True,
            "science_continuation_binding_required_for_future_science": True,
            "current_receipt_is_science_authority": False,
        }
    ):
        raise WorkhorseCanonicalRowError(
            "device-envelope projection does not recompute"
        )
    return value


def _envelope_row(
    envelope: Mapping[str, Any], *, B: int, temporal_k: int
) -> dict[str, Any]:
    _positive_int(B, "row B")
    _positive_int(temporal_k, "row temporal_k")
    rows = envelope.get("expected_rows")
    matches = (
        [
            row
            for row in rows
            if type(row) is dict
            and row.get("B") == B
            and row.get("temporal_k") == temporal_k
        ]
        if type(rows) is list
        else []
    )
    if len(matches) != 1:
        raise WorkhorseCanonicalRowError("row is absent or duplicated in envelope")
    return dict(matches[0])


def workhorse_row_checkpoint_binding_v2(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    science_continuation_contract_sha256: str,
    source_checkpoint_identity_sha256: str | None = None,
    checkpoint_ticks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bind provider-owned exact checkpoint/continuation boundaries."""
    science_digest = _sha(
        science_continuation_contract_sha256,
        "science continuation contract digest",
    )
    if source_checkpoint_identity_sha256 is not None:
        _sha(source_checkpoint_identity_sha256, "source checkpoint identity")
    target = contract.start_next_tick + contract.total_ticks
    ticks = (
        [target]
        if checkpoint_ticks is None
        else list(checkpoint_ticks)
    )
    if (
        not ticks
        or any(type(tick) is not int for tick in ticks)
        or ticks != sorted(set(ticks))
        or ticks[-1] != target
        or any(
            tick <= contract.start_next_tick
            or tick > target
            or (tick - contract.start_next_tick) % contract.temporal_k
            for tick in ticks
        )
    ):
        raise WorkhorseCanonicalRowError(
            "row checkpoint ticks must be unique acknowledged dispatch boundaries "
            "ending at the exact target"
        )
    body = {
        "schema_version": "eiel_workhorse_row_checkpoint_binding_v2",
        "checkpoint_schema": WORKHORSE_CHECKPOINT_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "absolute_start_next_tick": contract.start_next_tick,
        "absolute_target_next_tick": target,
        "checkpoint_ticks": ticks,
        "source_checkpoint_identity_sha256": source_checkpoint_identity_sha256,
        "science_continuation_contract_sha256": science_digest,
        "checkpoint_occurrence_owner": "OUTER_CANONICAL_CAMPAIGN_PROVIDER",
        "row_runner_publishes_checkpoint_occurrence": False,
        "science_continuation_identity_preserved": True,
    }
    return {**body, "checkpoint_binding_sha256": canonical_sha256(body)}


def _validate_checkpoint_binding(
    value: object, *, contract: FullEvolutionCudaWorkhorseContractV2
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "checkpoint_schema",
        "time_authority",
        "absolute_start_next_tick",
        "absolute_target_next_tick",
        "checkpoint_ticks",
        "source_checkpoint_identity_sha256",
        "science_continuation_contract_sha256",
        "checkpoint_occurrence_owner",
        "row_runner_publishes_checkpoint_occurrence",
        "science_continuation_identity_preserved",
        "checkpoint_binding_sha256",
    }
    binding = _exact_mapping(value, fields, "row checkpoint binding")
    expected = workhorse_row_checkpoint_binding_v2(
        contract,
        science_continuation_contract_sha256=str(
            binding["science_continuation_contract_sha256"]
        ),
        source_checkpoint_identity_sha256=binding["source_checkpoint_identity_sha256"],
        checkpoint_ticks=binding["checkpoint_ticks"],
    )
    if binding != expected:
        raise WorkhorseCanonicalRowError("row checkpoint binding changed")
    return binding


def _p10(values: Sequence[float]) -> float:
    if len(values) < 5 or any(
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in values
    ):
        raise WorkhorseCanonicalRowError("row requires at least five positive samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * 0.10
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise WorkhorseCanonicalRowError("row median requires positive samples")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _runtime_observation(
    *, free_bytes: int, total_device_bytes: int, device_uuid: str
) -> dict[str, Any]:
    _nonnegative_int(free_bytes, "runtime free bytes")
    _positive_int(total_device_bytes, "runtime total bytes")
    if free_bytes > total_device_bytes:
        raise WorkhorseCanonicalRowError("runtime free bytes exceed total")
    _trimmed(device_uuid, "runtime device UUID")
    body = {
        "schema_version": "eiel_workhorse_row_runtime_observation_v2",
        "device_uuid": device_uuid,
        "free_bytes": free_bytes,
        "total_device_bytes": total_device_bytes,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "CUDA_MEM_GET_INFO_UNDER_CANONICAL_EXCLUSIVE_PARENT",
    }
    return {**body, "observation_sha256": canonical_sha256(body)}


def _row_resource_conformance_v2(
    *,
    ledger: Mapping[str, Any],
    allocator_baseline_bytes: int,
    allocator_peak_bytes: int,
    free_vram_before_bytes: int,
    free_vram_after_bytes: int,
) -> tuple[bool, int, int]:
    """Recompute the two independent bounded-VRAM observations for one row."""
    allocator_baseline = _nonnegative_int(
        allocator_baseline_bytes, "row allocator baseline bytes"
    )
    allocator_peak = _nonnegative_int(allocator_peak_bytes, "row allocator peak bytes")
    free_before = _nonnegative_int(free_vram_before_bytes, "row free VRAM before bytes")
    free_after = _nonnegative_int(free_vram_after_bytes, "row free VRAM after bytes")
    if allocator_peak < allocator_baseline:
        raise WorkhorseCanonicalRowError("row allocator peak precedes its baseline")
    allocator_delta = allocator_peak - allocator_baseline
    observed_free_delta = max(0, free_before - free_after)
    allocator_allowance = _nonnegative_int(
        ledger.get("tensor_resident_bytes"), "row tensor resident bytes"
    ) + _nonnegative_int(
        ledger.get("allocator_owner_overhead_bytes"),
        "row allocator owner overhead bytes",
    )
    incremental_allowance = _nonnegative_int(
        ledger.get("incremental_device_bytes"), "row incremental device bytes"
    )
    return (
        allocator_delta <= allocator_allowance
        and observed_free_delta <= incremental_allowance,
        allocator_delta,
        observed_free_delta,
    )


def _validate_row_measurement_sample_v2(
    value: object,
    *,
    sample_index: int,
    window_index: int,
    dispatch_index_in_window: int,
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    committed_world_ticks: int,
    world_ticks_per_second: float,
    event_sink_receipt_index: int,
    event_sink_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    sample = _exact_mapping(
        value,
        {
            "sample_index",
            "window_index",
            "dispatch_index_in_window",
            "start_next_absolute_tick",
            "end_next_absolute_tick",
            "timing_source",
            "cuda_event_elapsed_ms",
            "event_canonicalization_tail_ms",
            "event_canonicalization_tail_scope",
            "committed_world_ticks",
            "world_ticks_per_second",
            "event_sink_receipt_index",
            "event_sink_receipt_path",
            "event_sink_receipt_file_sha256",
            "event_sink_receipt_body_sha256",
            "event_sink_dispatch_content_sha256",
            "measurement_sha256",
        },
        "row timing sample",
    )
    elapsed_ms = sample["cuda_event_elapsed_ms"]
    tail_ms = sample["event_canonicalization_tail_ms"]
    sample_body = {
        key: item for key, item in sample.items() if key != "measurement_sha256"
    }
    if (
        sample["sample_index"] != sample_index
        or sample["window_index"] != window_index
        or sample["dispatch_index_in_window"] != dispatch_index_in_window
        or sample["start_next_absolute_tick"] != start_next_absolute_tick
        or sample["end_next_absolute_tick"] != end_next_absolute_tick
        or end_next_absolute_tick <= start_next_absolute_tick
        or sample["timing_source"] != "cuda_event"
        or type(elapsed_ms) not in {int, float}
        or not math.isfinite(float(elapsed_ms))
        or float(elapsed_ms) <= 0
        or type(tail_ms) not in {int, float}
        or not math.isfinite(float(tail_ms))
        or float(tail_ms) < 0
        or sample["event_canonicalization_tail_scope"]
        != "POST_CUDA_EVENT_EXCLUDED_FROM_WORLD_TICKS_PER_SECOND"
        or sample["committed_world_ticks"] != committed_world_ticks
        or type(sample["world_ticks_per_second"]) not in {int, float}
        or not math.isfinite(float(sample["world_ticks_per_second"]))
    ):
        raise WorkhorseCanonicalRowError("row timing sample seal changed")
    expected_rate = committed_world_ticks * 1000.0 / float(elapsed_ms)
    if (
        not math.isclose(
            float(sample["world_ticks_per_second"]),
            expected_rate,
            rel_tol=_ROW_RATE_RELATIVE_TOLERANCE_V2,
            abs_tol=_ROW_RATE_ABSOLUTE_TOLERANCE_V2,
        )
        or not math.isclose(
            float(sample["world_ticks_per_second"]),
            float(world_ticks_per_second),
            rel_tol=_ROW_RATE_RELATIVE_TOLERANCE_V2,
            abs_tol=_ROW_RATE_ABSOLUTE_TOLERANCE_V2,
        )
        or sample["event_sink_receipt_index"] != event_sink_receipt_index
        or event_sink_receipt.get("start_next_absolute_tick")
        != start_next_absolute_tick
        or event_sink_receipt.get("end_next_absolute_tick") != end_next_absolute_tick
        or sample["event_sink_receipt_path"] != event_sink_receipt.get("receipt_path")
        or sample["event_sink_receipt_file_sha256"]
        != event_sink_receipt.get("receipt_file_sha256")
        or sample["event_sink_receipt_body_sha256"]
        != event_sink_receipt.get("receipt_body_sha256")
        or sample["event_sink_dispatch_content_sha256"]
        != event_sink_receipt.get("dispatch_content_sha256")
        or sample["measurement_sha256"] != canonical_sha256(sample_body)
    ):
        raise WorkhorseCanonicalRowError("row timing sample seal changed")
    for field in (
        "event_sink_receipt_file_sha256",
        "event_sink_receipt_body_sha256",
        "event_sink_dispatch_content_sha256",
    ):
        _sha(sample[field], f"row timing sample {field}")
    return sample


def _validate_row_measurement_window_v3(
    value: object,
    *,
    window_index: int,
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    measurement_window_ticks: int,
    dispatch_measurement_start_index: int,
    committed_world_ticks: int,
    dispatch_measurements: Sequence[Mapping[str, Any]],
    world_ticks_per_second: float,
) -> dict[str, Any]:
    window = _exact_mapping(
        value,
        {
            "window_index",
            "start_next_absolute_tick",
            "end_next_absolute_tick",
            "measurement_window_ticks",
            "dispatch_measurement_start_index",
            "dispatch_measurement_count",
            "cuda_event_elapsed_ms_sum",
            "committed_world_ticks",
            "world_ticks_per_second",
            "dispatch_measurement_sha256s",
            "window_measurement_sha256",
        },
        "fixed-tick measurement window",
    )
    elapsed_values = [
        float(row["cuda_event_elapsed_ms"]) for row in dispatch_measurements
    ]
    elapsed_sum = math.fsum(elapsed_values)
    expected_rate = committed_world_ticks * 1000.0 / elapsed_sum
    body = {
        key: item for key, item in window.items() if key != "window_measurement_sha256"
    }
    if (
        window["window_index"] != window_index
        or window["start_next_absolute_tick"] != start_next_absolute_tick
        or window["end_next_absolute_tick"] != end_next_absolute_tick
        or end_next_absolute_tick - start_next_absolute_tick != measurement_window_ticks
        or window["measurement_window_ticks"] != measurement_window_ticks
        or window["dispatch_measurement_start_index"]
        != dispatch_measurement_start_index
        or window["dispatch_measurement_count"] != len(dispatch_measurements)
        or window["committed_world_ticks"] != committed_world_ticks
        or type(window["cuda_event_elapsed_ms_sum"]) not in {int, float}
        or not math.isclose(
            float(window["cuda_event_elapsed_ms_sum"]),
            elapsed_sum,
            rel_tol=_ROW_RATE_RELATIVE_TOLERANCE_V2,
            abs_tol=_ROW_RATE_ABSOLUTE_TOLERANCE_V2,
        )
        or type(window["world_ticks_per_second"]) not in {int, float}
        or not math.isclose(
            float(window["world_ticks_per_second"]),
            expected_rate,
            rel_tol=_ROW_RATE_RELATIVE_TOLERANCE_V2,
            abs_tol=_ROW_RATE_ABSOLUTE_TOLERANCE_V2,
        )
        or not math.isclose(
            float(window["world_ticks_per_second"]),
            float(world_ticks_per_second),
            rel_tol=_ROW_RATE_RELATIVE_TOLERANCE_V2,
            abs_tol=_ROW_RATE_ABSOLUTE_TOLERANCE_V2,
        )
        or window["dispatch_measurement_sha256s"]
        != [row["measurement_sha256"] for row in dispatch_measurements]
        or window["window_measurement_sha256"] != canonical_sha256(body)
    ):
        raise WorkhorseCanonicalRowError("fixed-tick measurement window changed")
    return window


def _reopen_row_event_sink_receipts_v2(
    values: Sequence[object],
) -> list[dict[str, Any]]:
    reopened: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise WorkhorseCanonicalRowError(
                f"row event-sink receipt {index} is malformed"
            )
        path = value.get("receipt_path")
        if not isinstance(path, str) or not path:
            raise WorkhorseCanonicalRowError(
                f"row event-sink receipt {index} lacks its path"
            )
        try:
            current = validate_workhorse_event_sink_receipt_v2(path)
        except Exception as exc:
            raise WorkhorseCanonicalRowError(
                f"row event-sink receipt {index} did not reopen: {exc}"
            ) from exc
        acknowledged = dict(value)
        # Early sparse-v3 row receipts predate the validator's derived
        # raw-payload-retention projection.  The sealed sink receipt and its
        # payload were already bound by the acknowledged file/body/content
        # hashes; normalize only this omitted, recomputable V3 projection.
        if (
            current.get("schema_version")
            == WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3
            and current.get("raw_payload_retained") is True
            and "raw_payload_retained" not in acknowledged
        ):
            acknowledged["raw_payload_retained"] = True
        if current != acknowledged:
            raise WorkhorseCanonicalRowError(
                f"row event-sink receipt {index} changed after acknowledgement"
            )
        if current.get("schema_version") not in {
            WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3,
            WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
        }:
            raise WorkhorseCanonicalRowError(
                f"row event-sink receipt {index} is not an accepted sparse transport"
            )
        reopened.append(current)
    return reopened


def _seal_row_execution_result(
    output_path: str | Path,
    *,
    envelope: Mapping[str, Any],
    contract: FullEvolutionCudaWorkhorseContractV2,
    parent: Mapping[str, Any],
    semantic_binding: Mapping[str, Any],
    micro_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    resource_ledger: Mapping[str, Any],
    planned_admission: Mapping[str, Any],
    runtime_admission: Mapping[str, Any],
    runtime_observation: Mapping[str, Any],
    outcome: str,
    measurement_schedule: Mapping[str, Any],
    samples: Sequence[float],
    dispatch_measurements: Sequence[Mapping[str, Any]],
    measurement_windows: Sequence[Mapping[str, Any]],
    event_sink_receipts: Sequence[Mapping[str, Any]],
    resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    if outcome not in {
        "MEASURED",
        "EVENT_PAGE_OVERFLOW_RECOVERED_NONSELECTABLE",
        "RESOURCE_LEDGER_REJECTED",
        "RUNTIME_HEADROOM_REJECTED",
    }:
        raise WorkhorseCanonicalRowError("row outcome is unsupported")
    target = contract.start_next_tick + contract.total_ticks
    rates = [float(value) for value in samples]
    if outcome == "RUNTIME_HEADROOM_REJECTED":
        if rates or dispatch_measurements or measurement_windows or event_sink_receipts:
            raise WorkhorseCanonicalRowError(
                "runtime-headroom rejection cannot claim execution samples"
            )
        p10 = None
        minimum = None
        median = None
        committed_next_tick = contract.start_next_tick
        if runtime_admission.get("admitted") is not False:
            raise WorkhorseCanonicalRowError(
                "runtime-headroom rejection requires a negative live admission"
            )
    else:
        p10 = _p10(rates)
        minimum = min(rates)
        median = _median(rates)
        committed_next_tick = target
        if (
            runtime_admission.get("admitted") is not True
            or len(rates) != measurement_schedule["measured_windows"]
            or len(measurement_windows) != measurement_schedule["measured_windows"]
            or len(dispatch_measurements) != measurement_schedule["measured_dispatches"]
            or len(event_sink_receipts)
            != measurement_schedule["warmup_dispatches"]
            + measurement_schedule["measured_dispatches"]
        ):
            raise WorkhorseCanonicalRowError("measured row coverage is incomplete")
    body = {
        "schema_version": WORKHORSE_CANONICAL_ROW_EXECUTION_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "envelope_binding": {
            "receipt_path": envelope["receipt_path"],
            "receipt_file_sha256": envelope["receipt_file_sha256"],
            "receipt_content_sha256": envelope["receipt_content_sha256"],
        },
        "contract": contract.to_dict(),
        "canonical_parent_authority_binding": dict(parent),
        "artifact": envelope["artifact"],
        "source_identity": envelope["source_identity"],
        "source_identity_sha256": envelope["source_identity_sha256"],
        "device_identity": envelope["device_identity"],
        "semantic_binding": dict(semantic_binding),
        "micro_denominator_binding": dict(micro_binding),
        "checkpoint_binding": dict(checkpoint_binding),
        "resource_ledger": dict(resource_ledger),
        "planned_admission": dict(planned_admission),
        "runtime_admission": dict(runtime_admission),
        "runtime_observation": dict(runtime_observation),
        "outcome": outcome,
        "absolute_frontiers": {
            "start_next_tick": contract.start_next_tick,
            "target_next_tick": target,
            "committed_next_tick": committed_next_tick,
        },
        "measurement_schedule": dict(measurement_schedule),
        "timing_source": "cuda_event_summed_fixed_absolute_tick_window_v3",
        "sink_boundary": "external_exact_content_ack_outside_timed_interval",
        "fixed_window_world_ticks_per_second_samples": rates,
        "dispatch_measurements": [dict(row) for row in dispatch_measurements],
        "measurement_windows": [dict(row) for row in measurement_windows],
        "p10_world_ticks_per_second": p10,
        "minimum_fixed_window_world_ticks_per_second": minimum,
        "median_fixed_window_world_ticks_per_second": median,
        "event_sink_receipts": [dict(row) for row in event_sink_receipts],
        "resource_observation": dict(resource_observation),
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_canonical_row_execution_result_v2(output)


def validate_workhorse_canonical_row_execution_result_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_CANONICAL_ROW_EXECUTION_SCHEMA_V2,
        label="Workhorse canonical row execution result",
    )
    _reject_generation_authority(value, path="row_execution")
    if (
        value.get("claim_class") != _CLAIM_CLASS
        or any(value.get(field) is not False for field in _NO_AUTHORITY)
        or value.get("timing_source")
        != "cuda_event_summed_fixed_absolute_tick_window_v3"
        or value.get("sink_boundary")
        != "external_exact_content_ack_outside_timed_interval"
    ):
        raise WorkhorseCanonicalRowError("row execution authority/timing flags changed")
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        value.get("contract", {})
    )
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        value.get("canonical_parent_authority_binding"),
        expected_operation="EXECUTION_ROW",
    )
    if parent["contract"] != contract.to_dict():
        raise WorkhorseCanonicalRowError("row execution differs from parent contract")
    schedule = validate_workhorse_fixed_tick_measurement_schedule_v3(
        value.get("measurement_schedule"), temporal_k=contract.temporal_k
    )
    if contract.total_ticks != schedule["total_ticks"]:
        raise WorkhorseCanonicalRowError(
            "row contract differs from its fixed-tick measurement schedule"
        )
    envelope_binding = _exact_mapping(
        value.get("envelope_binding"),
        {"receipt_path", "receipt_file_sha256", "receipt_content_sha256"},
        "row envelope binding",
    )
    _sha(envelope_binding["receipt_file_sha256"], "row envelope file digest")
    _sha(envelope_binding["receipt_content_sha256"], "row envelope content digest")
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_binding["receipt_path"]
    )
    if (
        envelope["receipt_file_sha256"] != envelope_binding["receipt_file_sha256"]
        or envelope["receipt_content_sha256"]
        != envelope_binding["receipt_content_sha256"]
        or value.get("artifact") != envelope["artifact"]
        or value.get("source_identity") != envelope["source_identity"]
        or value.get("source_identity_sha256") != envelope["source_identity_sha256"]
        or value.get("device_identity") != envelope["device_identity"]
    ):
        raise WorkhorseCanonicalRowError("row execution changed envelope identity")
    _validate_parent_for_artifact(parent, envelope["artifact"])
    row = _envelope_row(envelope, B=contract.B, temporal_k=contract.temporal_k)
    if value.get("planned_admission") != row["planned_admission"]:
        raise WorkhorseCanonicalRowError("row planned admission changed")
    components = _owner_components(
        envelope["owner_components_by_k"][str(contract.temporal_k)],
        backend_id=contract.backend_id,
        temporal_k=contract.temporal_k,
    )
    ledger = estimate_workhorse_resources_v2(
        contract, owner_components=components
    ).to_dict()
    if value.get("resource_ledger") != ledger:
        raise WorkhorseCanonicalRowError("row resource ledger changed")
    checkpoint = _validate_checkpoint_binding(
        value.get("checkpoint_binding"), contract=contract
    )
    if (
        checkpoint["source_checkpoint_identity_sha256"]
        != parent["source_checkpoint_identity_sha256"]
    ):
        raise WorkhorseCanonicalRowError("row restore checkpoint identity changed")
    runtime = value.get("runtime_observation")
    if type(runtime) is not dict:
        raise WorkhorseCanonicalRowError("row runtime observation is missing")
    runtime_body = {
        key: item for key, item in runtime.items() if key != "observation_sha256"
    }
    if (
        runtime.get("schema_version") != "eiel_workhorse_row_runtime_observation_v2"
        or runtime.get("source") != "CUDA_MEM_GET_INFO_UNDER_CANONICAL_EXCLUSIVE_PARENT"
        or runtime.get("device_uuid") != envelope["device_identity"]["uuid"]
        or runtime.get("total_device_bytes")
        != envelope["device_identity"]["total_memory_bytes"]
        or runtime.get("observation_sha256") != canonical_sha256(runtime_body)
    ):
        raise WorkhorseCanonicalRowError("row runtime observation changed identity")
    free_bytes = _nonnegative_int(runtime.get("free_bytes"), "row runtime free bytes")
    admission = admit_workhorse_vram_v2(
        contract,
        free_bytes=free_bytes,
        total_device_bytes=runtime["total_device_bytes"],
        owner_components=components,
        reserve_bytes=envelope["capacity_basis"]["reserve_bytes_requested"],
    ).to_dict()
    if value.get("runtime_admission") != admission:
        raise WorkhorseCanonicalRowError("row runtime admission changed")
    outcome = value.get("outcome")
    samples = value.get("fixed_window_world_ticks_per_second_samples")
    dispatch_measurements = value.get("dispatch_measurements")
    measurement_windows = value.get("measurement_windows")
    sink_receipts = value.get("event_sink_receipts")
    if (
        type(samples) is not list
        or type(dispatch_measurements) is not list
        or type(measurement_windows) is not list
        or type(sink_receipts) is not list
    ):
        raise WorkhorseCanonicalRowError("row sample/sink collections are malformed")
    target = contract.start_next_tick + contract.total_ticks
    frontiers = value.get("absolute_frontiers")
    if outcome == "RUNTIME_HEADROOM_REJECTED":
        if (
            admission["admitted"] is not False
            or samples
            or dispatch_measurements
            or measurement_windows
            or sink_receipts
            or value.get("p10_world_ticks_per_second") is not None
            or value.get("minimum_fixed_window_world_ticks_per_second") is not None
            or value.get("median_fixed_window_world_ticks_per_second") is not None
            or frontiers
            != {
                "start_next_tick": contract.start_next_tick,
                "target_next_tick": target,
                "committed_next_tick": contract.start_next_tick,
            }
        ):
            raise WorkhorseCanonicalRowError(
                "runtime-headroom rejection is inconsistent"
            )
    elif outcome in {
        "MEASURED",
        "EVENT_PAGE_OVERFLOW_RECOVERED_NONSELECTABLE",
        "RESOURCE_LEDGER_REJECTED",
    }:
        p10 = _p10(samples)
        if (
            admission["admitted"] is not True
            or len(samples) != schedule["measured_windows"]
            or len(measurement_windows) != schedule["measured_windows"]
            or len(dispatch_measurements) != schedule["measured_dispatches"]
            or len(sink_receipts)
            != schedule["warmup_dispatches"] + schedule["measured_dispatches"]
            or value.get("p10_world_ticks_per_second") != p10
            or value.get("minimum_fixed_window_world_ticks_per_second") != min(samples)
            or value.get("median_fixed_window_world_ticks_per_second")
            != _median(samples)
            or frontiers
            != {
                "start_next_tick": contract.start_next_tick,
                "target_next_tick": target,
                "committed_next_tick": target,
            }
        ):
            raise WorkhorseCanonicalRowError("measured row coverage is inconsistent")
        reopened_sinks = _reopen_row_event_sink_receipts_v2(sink_receipts)
        if outcome == "MEASURED" and any(
            sink.get("benchmark_selectable") is not True for sink in reopened_sinks
        ):
            raise WorkhorseCanonicalRowError(
                "measured row contains an overflow-recovered dispatch"
            )
        if outcome == "EVENT_PAGE_OVERFLOW_RECOVERED_NONSELECTABLE" and all(
            sink.get("benchmark_selectable") is True for sink in reopened_sinks
        ):
            raise WorkhorseCanonicalRowError(
                "event-overflow rejection lacks a recovered dispatch"
            )
        dispatches_per_window = schedule["dispatches_per_measurement_window"]
        measurement_start_tick = contract.start_next_tick + schedule["warmup_ticks"]
        for index, sample in enumerate(dispatch_measurements):
            window_index = index // dispatches_per_window
            dispatch_index_in_window = index % dispatches_per_window
            start_tick = measurement_start_tick + index * contract.temporal_k
            end_tick = start_tick + contract.temporal_k
            event_sink_receipt_index = schedule["warmup_dispatches"] + index
            _validate_row_measurement_sample_v2(
                sample,
                sample_index=index,
                window_index=window_index,
                dispatch_index_in_window=dispatch_index_in_window,
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=end_tick,
                committed_world_ticks=contract.B * contract.temporal_k,
                world_ticks_per_second=sample["world_ticks_per_second"],
                event_sink_receipt_index=event_sink_receipt_index,
                event_sink_receipt=reopened_sinks[event_sink_receipt_index],
            )
        for window_index, window in enumerate(measurement_windows):
            dispatch_start = window_index * dispatches_per_window
            dispatch_end = dispatch_start + dispatches_per_window
            window_dispatches = dispatch_measurements[dispatch_start:dispatch_end]
            start_tick = (
                measurement_start_tick
                + window_index * schedule["measurement_window_ticks"]
            )
            _validate_row_measurement_window_v3(
                window,
                window_index=window_index,
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=(
                    start_tick + schedule["measurement_window_ticks"]
                ),
                measurement_window_ticks=schedule["measurement_window_ticks"],
                dispatch_measurement_start_index=dispatch_start,
                committed_world_ticks=(
                    contract.B * schedule["measurement_window_ticks"]
                ),
                dispatch_measurements=window_dispatches,
                world_ticks_per_second=samples[window_index],
            )
    else:
        raise WorkhorseCanonicalRowError("row execution outcome is unsupported")
    if (
        type(value.get("semantic_binding")) is not dict
        or type(value.get("micro_denominator_binding")) is not dict
    ):
        raise WorkhorseCanonicalRowError("row prerequisite bindings are absent")
    if type(value.get("resource_observation")) is not dict:
        raise WorkhorseCanonicalRowError("row resource evidence is absent")
    resource_observation = value["resource_observation"]
    runtime_inspection = resource_observation.get("runtime_inspection")
    if (
        type(runtime_inspection) is not dict
        or resource_observation.get("runtime_inspection_sha256")
        != canonical_sha256(runtime_inspection)
        or runtime_inspection.get("backend_id") != contract.backend_id
        or runtime_inspection.get("profile_id") != contract.profile_id
        or runtime_inspection.get("route_id") != contract.route_id
        or str(runtime_inspection.get("device_uuid", "")).replace("-", "").lower()
        != str(envelope["device_identity"]["uuid"]).replace("-", "").lower()
    ):
        raise WorkhorseCanonicalRowError("row runtime inspection changed identity")
    if outcome == "RUNTIME_HEADROOM_REJECTED":
        if (
            set(resource_observation)
            != {
                "runtime_inspection_sha256",
                "runtime_inspection",
                "rejection_reason_code",
            }
            or resource_observation.get("rejection_reason_code") != "WAIT_VRAM_HEADROOM"
        ):
            raise WorkhorseCanonicalRowError(
                "row runtime-headroom resource evidence is inconsistent"
            )
    else:
        if set(resource_observation) != {
            "runtime_inspection_sha256",
            "runtime_inspection",
            "graph_inspection",
            "allocator_baseline_bytes",
            "allocator_peak_bytes",
            "allocator_peak_delta_bytes",
            "free_vram_after_bytes",
            "observed_free_vram_delta_bytes",
            "resource_ledger_conforms",
        }:
            raise WorkhorseCanonicalRowError("row resource evidence fields changed")
        allocator_baseline = _nonnegative_int(
            resource_observation.get("allocator_baseline_bytes"),
            "row allocator baseline bytes",
        )
        allocator_peak = _nonnegative_int(
            resource_observation.get("allocator_peak_bytes"),
            "row allocator peak bytes",
        )
        allocator_delta = _nonnegative_int(
            resource_observation.get("allocator_peak_delta_bytes"),
            "row allocator peak delta bytes",
        )
        free_after = _nonnegative_int(
            resource_observation.get("free_vram_after_bytes"),
            "row free VRAM after bytes",
        )
        observed_free_delta = _nonnegative_int(
            resource_observation.get("observed_free_vram_delta_bytes"),
            "row observed free VRAM delta bytes",
        )
        if (
            allocator_peak < allocator_baseline
            or allocator_delta != allocator_peak - allocator_baseline
            or free_after > runtime["total_device_bytes"]
            or observed_free_delta != max(0, free_bytes - free_after)
        ):
            raise WorkhorseCanonicalRowError("row resource arithmetic changed")
        expected_resource_conforms, expected_allocator_delta, expected_free_delta = (
            _row_resource_conformance_v2(
                ledger=ledger,
                allocator_baseline_bytes=allocator_baseline,
                allocator_peak_bytes=allocator_peak,
                free_vram_before_bytes=free_bytes,
                free_vram_after_bytes=free_after,
            )
        )
        if (
            allocator_delta != expected_allocator_delta
            or observed_free_delta != expected_free_delta
            or resource_observation.get("resource_ledger_conforms")
            is not expected_resource_conforms
            or (
                outcome
                in {
                    "MEASURED",
                    "EVENT_PAGE_OVERFLOW_RECOVERED_NONSELECTABLE",
                }
            )
            is not expected_resource_conforms
        ):
            raise WorkhorseCanonicalRowError(
                "row resource-ledger verdict does not recompute"
            )
    return value


def _parent_row_contract(
    parent: Mapping[str, Any], *, envelope: Mapping[str, Any], B: int, temporal_k: int
) -> FullEvolutionCudaWorkhorseContractV2:
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(parent["contract"])
    if contract.B != B or contract.temporal_k != temporal_k:
        raise WorkhorseCanonicalRowError("parent Attempt binds another B/K row")
    row = _envelope_row(envelope, B=B, temporal_k=temporal_k)
    if row["planned_admission"].get("admitted") is not True:
        raise WorkhorseCanonicalRowError(
            "single-row runner accepts only envelope-admitted rows"
        )
    return contract


def run_workhorse_canonical_admitted_row_v2(
    artifact_path: str | Path,
    *,
    envelope_receipt_path: str | Path,
    canonical_parent_authority_binding: Mapping[str, Any],
    authoritative_micro_receipt_path: str | Path,
    semantic_gate_receipt_path: str | Path,
    science_continuation_contract_sha256: str,
    event_sink_root: str | Path,
    execution_result_path: str | Path,
    B: int,
    temporal_k: int,
    world_identity_binding: Mapping[str, Any] | None = None,
    warmup_ticks: int = 192,
    measurement_window_ticks: int = 64,
    measured_windows: int = 20,
    checkpoint_transport: Any | None = None,
    checkpoint_ticks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run one admitted row under one already-canonical outer Attempt.

    Scheduler admission, heartbeat, and terminalization remain outer-provider
    duties.  CUDA-event timing encloses only the transition call; durable event
    publication and acknowledgement occur after the end event.
    """
    schedule = workhorse_fixed_tick_measurement_schedule_v3(
        temporal_k=temporal_k,
        warmup_ticks=warmup_ticks,
        measurement_window_ticks=measurement_window_ticks,
        measured_windows=measured_windows,
    )
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_receipt_path
    )
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation="EXECUTION_ROW",
    )
    contract = _parent_row_contract(
        parent, envelope=envelope, B=B, temporal_k=temporal_k
    )
    world_identity = validate_workhorse_world_identity_binding_v2(
        (
            workhorse_world_identity_binding_v2(
                world_uids=tuple(210_000 + world for world in range(B)),
                rng_keys=(83,) * B,
            )
            if world_identity_binding is None
            else world_identity_binding
        ),
        expected_world_count=B,
    )
    if contract.total_ticks != schedule["total_ticks"]:
        raise WorkhorseCanonicalRowError(
            "row contract total_ticks must exactly cover the fixed-tick schedule"
        )
    artifact_binding = workhorse_artifact_binding_v2(artifact_path)
    if artifact_binding != envelope["artifact"]:
        raise WorkhorseCanonicalRowError("row artifact differs from envelope")
    _validate_parent_for_artifact(parent, artifact_binding)

    # Runtime imports are intentionally delayed so host/OFF paths have no CUDA
    # or torch import cost.
    from .authority import (
        validate_authoritative_micro_denominator_v2,
        validate_workhorse_semantic_gate_receipt_v2,
    )
    from .probe import (
        _acknowledge_events,
        _advance,
        _allocate_owner,
        _initialize_owner_exact,
        _load,
        _prepare_dispatch_v2,
        _prepare_graph_if_needed,
        _release_graph,
        _torch,
    )
    from ..benchmark import cuda_device_identity_v2

    torch = _torch()
    loaded = _load(artifact_path, torch)
    runtime = dict(loaded.module.runtime_inspect())
    device_identity = cuda_device_identity_v2(torch, torch.cuda.current_device())
    if device_identity != envelope["device_identity"]:
        raise WorkhorseCanonicalRowError("row runtime device differs from envelope")
    semantic = validate_workhorse_semantic_gate_receipt_v2(
        semantic_gate_receipt_path,
        expected_shape_id=contract.shape_id,
        expected_candidate_artifact_sha256=artifact_binding["sha256"],
        expected_candidate_backend_id=contract.backend_id,
    )
    micro = validate_authoritative_micro_denominator_v2(
        authoritative_micro_receipt_path,
        expected_device_identity=device_identity,
    )
    components = _owner_components(
        envelope["owner_components_by_k"][str(temporal_k)],
        backend_id=contract.backend_id,
        temporal_k=temporal_k,
    )
    ledger = estimate_workhorse_resources_v2(contract, owner_components=components)
    free_before, total_observed = torch.cuda.mem_get_info(torch.cuda.current_device())
    observation = _runtime_observation(
        free_bytes=int(free_before),
        total_device_bytes=int(total_observed),
        device_uuid=str(device_identity["uuid"]),
    )
    if total_observed != device_identity["total_memory_bytes"]:
        raise WorkhorseCanonicalRowError("row runtime total VRAM changed device")
    runtime_admission = admit_workhorse_vram_v2(
        contract,
        free_bytes=int(free_before),
        total_device_bytes=int(total_observed),
        owner_components=components,
        reserve_bytes=envelope["capacity_basis"]["reserve_bytes_requested"],
    )
    planned = _envelope_row(envelope, B=B, temporal_k=temporal_k)["planned_admission"]
    checkpoint_binding = workhorse_row_checkpoint_binding_v2(
        contract,
        science_continuation_contract_sha256=science_continuation_contract_sha256,
        source_checkpoint_identity_sha256=parent["source_checkpoint_identity_sha256"],
        checkpoint_ticks=checkpoint_ticks,
    )
    if (
        science_continuation_contract_sha256
        != parent["science_continuation_contract_sha256"]
    ):
        raise WorkhorseCanonicalRowError(
            "row science continuation identity differs from parent Attempt"
        )
    if not runtime_admission.admitted:
        return _seal_row_execution_result(
            execution_result_path,
            envelope=envelope,
            contract=contract,
            parent=parent,
            semantic_binding=semantic,
            micro_binding=micro,
            checkpoint_binding=checkpoint_binding,
            resource_ledger=ledger.to_dict(),
            planned_admission=planned,
            runtime_admission=runtime_admission.to_dict(),
            runtime_observation=observation,
            outcome="RUNTIME_HEADROOM_REJECTED",
            measurement_schedule=schedule,
            samples=[],
            dispatch_measurements=[],
            measurement_windows=[],
            event_sink_receipts=[],
            resource_observation={
                "runtime_inspection_sha256": canonical_sha256(runtime),
                "runtime_inspection": runtime,
                "rejection_reason_code": "WAIT_VRAM_HEADROOM",
            },
        )

    allocator_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    owner = None
    graph_inspection = None
    sink_receipts: list[dict[str, Any]] = []
    samples: list[float] = []
    dispatch_measurement_rows: list[dict[str, Any]] = []
    measurement_window_rows: list[dict[str, Any]] = []
    checkpoint_schedule = list(checkpoint_binding["checkpoint_ticks"])
    checkpoint_index = 0
    try:
        owner = _allocate_owner(torch, loaded, B, temporal_k)
        owner.attempt_authority_binding = parent
        _initialize_owner_exact(
            torch,
            loaded,
            owner,
            world_identity_binding=world_identity,
            start_tick=contract.start_next_tick,
            fixture=False,
        )
        graph_inspection = _prepare_graph_if_needed(loaded, owner, temporal_k, 0)

        def capture_acknowledged_checkpoint(ack: Mapping[str, Any]) -> None:
            nonlocal checkpoint_index
            if checkpoint_transport is None:
                return
            if (
                checkpoint_index < len(checkpoint_schedule)
                and checkpoint_schedule[checkpoint_index] < owner.committed_next_tick
            ):
                raise WorkhorseCanonicalRowError(
                    "row passed an exact checkpoint boundary without capture"
                )
            if (
                checkpoint_index < len(checkpoint_schedule)
                and checkpoint_schedule[checkpoint_index] == owner.committed_next_tick
            ):
                checkpoint_transport.capture_workhorse_checkpoint(
                    owner.state,
                    contract=contract,
                    checkpoint_at_tick=owner.committed_next_tick,
                    event_sink_receipt_path=ack["receipt_path"],
                )
                checkpoint_index += 1

        for warmup in range(schedule["warmup_dispatches"]):
            _advance(loaded, owner, temporal_k, 0)
            ack = _acknowledge_events(
                owner,
                torch,
                sink_root=Path(event_sink_root)
                / f"b{B}-k{temporal_k}-warmup-{warmup}",
                sink_id="workhorse-canonical-row-warmup-v2",
                retain_sink_payload=False,
            )
            sink_receipts.append(ack)
            capture_acknowledged_checkpoint(ack)
        sample_index = 0
        for window_index in range(schedule["measured_windows"]):
            window_start_tick = owner.committed_next_tick
            window_dispatch_start = sample_index
            window_dispatches: list[dict[str, Any]] = []
            for dispatch_index_in_window in range(
                schedule["dispatches_per_measurement_window"]
            ):
                dispatch_start_tick = owner.committed_next_tick
                _prepare_dispatch_v2(loaded, owner, temporal_k, 0)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _advance(loaded, owner, temporal_k, 0)
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise WorkhorseCanonicalRowError(
                        "CUDA event returned a non-positive row duration"
                    )
                rate = B * temporal_k * 1000.0 / elapsed_ms
                ack = _acknowledge_events(
                    owner,
                    torch,
                    sink_root=Path(event_sink_root)
                    / f"b{B}-k{temporal_k}-sample-{sample_index}",
                    sink_id="workhorse-canonical-row-measurement-v3",
                    retain_sink_payload=False,
                )
                raw_tail_ms = owner.last_event_canonicalization_tail_ms
                if type(raw_tail_ms) not in {int, float}:
                    raise WorkhorseCanonicalRowError(
                        "event canonicalization tail is absent after acknowledgement"
                    )
                canonicalization_tail_ms = float(raw_tail_ms)
                if (
                    not math.isfinite(canonicalization_tail_ms)
                    or canonicalization_tail_ms < 0
                ):
                    raise WorkhorseCanonicalRowError(
                        "event canonicalization tail is not finite and nonnegative"
                    )
                sink_receipts.append(ack)
                capture_acknowledged_checkpoint(ack)
                sample_body = {
                    "sample_index": sample_index,
                    "window_index": window_index,
                    "dispatch_index_in_window": dispatch_index_in_window,
                    "start_next_absolute_tick": dispatch_start_tick,
                    "end_next_absolute_tick": dispatch_start_tick + temporal_k,
                    "timing_source": "cuda_event",
                    "cuda_event_elapsed_ms": elapsed_ms,
                    "event_canonicalization_tail_ms": canonicalization_tail_ms,
                    "event_canonicalization_tail_scope": (
                        "POST_CUDA_EVENT_EXCLUDED_FROM_WORLD_TICKS_PER_SECOND"
                    ),
                    "committed_world_ticks": B * temporal_k,
                    "world_ticks_per_second": rate,
                    "event_sink_receipt_index": (
                        schedule["warmup_dispatches"] + sample_index
                    ),
                    "event_sink_receipt_path": ack["receipt_path"],
                    "event_sink_receipt_file_sha256": ack["receipt_file_sha256"],
                    "event_sink_receipt_body_sha256": ack["receipt_body_sha256"],
                    "event_sink_dispatch_content_sha256": ack[
                        "dispatch_content_sha256"
                    ],
                }
                sealed_sample = {
                    **sample_body,
                    "measurement_sha256": canonical_sha256(sample_body),
                }
                dispatch_measurement_rows.append(sealed_sample)
                window_dispatches.append(sealed_sample)
                sample_index += 1
            window_elapsed_ms = math.fsum(
                float(row["cuda_event_elapsed_ms"]) for row in window_dispatches
            )
            window_rate = (
                B * schedule["measurement_window_ticks"] * 1000.0 / window_elapsed_ms
            )
            window_body = {
                "window_index": window_index,
                "start_next_absolute_tick": window_start_tick,
                "end_next_absolute_tick": owner.committed_next_tick,
                "measurement_window_ticks": schedule["measurement_window_ticks"],
                "dispatch_measurement_start_index": window_dispatch_start,
                "dispatch_measurement_count": len(window_dispatches),
                "cuda_event_elapsed_ms_sum": window_elapsed_ms,
                "committed_world_ticks": (B * schedule["measurement_window_ticks"]),
                "world_ticks_per_second": window_rate,
                "dispatch_measurement_sha256s": [
                    row["measurement_sha256"] for row in window_dispatches
                ],
            }
            samples.append(window_rate)
            measurement_window_rows.append(
                {
                    **window_body,
                    "window_measurement_sha256": canonical_sha256(window_body),
                }
            )
        allocator_peak = int(torch.cuda.max_memory_allocated())
        if checkpoint_transport is not None:
            if (
                owner.committed_next_tick
                != contract.start_next_tick + contract.total_ticks
                or not sink_receipts
                or checkpoint_index != len(checkpoint_schedule)
            ):
                raise WorkhorseCanonicalRowError(
                    "row ACS checkpoints lack their exact acknowledged frontiers"
                )
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    free_after, total_after = torch.cuda.mem_get_info(torch.cuda.current_device())
    if total_after != total_observed:
        raise WorkhorseCanonicalRowError("row device total changed during execution")
    resource_conforms, allocator_delta, observed_free_delta = (
        _row_resource_conformance_v2(
            ledger=ledger.to_dict(),
            allocator_baseline_bytes=allocator_before,
            allocator_peak_bytes=allocator_peak,
            free_vram_before_bytes=int(free_before),
            free_vram_after_bytes=int(free_after),
        )
    )
    transport_selectable = all(
        receipt.get("benchmark_selectable") is True for receipt in sink_receipts
    )
    outcome = (
        "RESOURCE_LEDGER_REJECTED"
        if not resource_conforms
        else (
            "MEASURED"
            if transport_selectable
            else "EVENT_PAGE_OVERFLOW_RECOVERED_NONSELECTABLE"
        )
    )
    return _seal_row_execution_result(
        execution_result_path,
        envelope=envelope,
        contract=contract,
        parent=parent,
        semantic_binding=semantic,
        micro_binding=micro,
        checkpoint_binding=checkpoint_binding,
        resource_ledger=ledger.to_dict(),
        planned_admission=planned,
        runtime_admission=runtime_admission.to_dict(),
        runtime_observation=observation,
        outcome=outcome,
        measurement_schedule=schedule,
        samples=samples,
        dispatch_measurements=dispatch_measurement_rows,
        measurement_windows=measurement_window_rows,
        event_sink_receipts=sink_receipts,
        resource_observation={
            "runtime_inspection": runtime,
            "runtime_inspection_sha256": canonical_sha256(runtime),
            "graph_inspection": graph_inspection,
            "allocator_baseline_bytes": allocator_before,
            "allocator_peak_bytes": allocator_peak,
            "allocator_peak_delta_bytes": allocator_delta,
            "free_vram_after_bytes": int(free_after),
            "observed_free_vram_delta_bytes": observed_free_delta,
            "resource_ledger_conforms": resource_conforms,
        },
    )


def validate_workhorse_canonical_terminal_binding_v2(
    value: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = _exact_mapping(
        value,
        {
            "schema_version",
            "authority_source",
            "status",
            "attempt_terminal_status",
            "run_id",
            "attempt_id",
            "terminal_bundle_path",
            "terminal_bundle_file_sha256",
            "terminal_bundle_digest",
            "final_frontiers",
            "binding_sha256",
        },
        "canonical terminal binding",
    )
    if (
        terminal["schema_version"] != WORKHORSE_CANONICAL_TERMINAL_BINDING_SCHEMA_V2
        or terminal["authority_source"] != _PARENT_SOURCE
        or terminal["status"] != "TERMINAL"
        or terminal["attempt_terminal_status"] not in {"COMPLETED", "FAILED"}
        or terminal["run_id"] != parent["run_id"]
        or terminal["attempt_id"] != parent["attempt_id"]
    ):
        raise WorkhorseCanonicalRowError("canonical terminal identity changed")
    _sha(terminal["terminal_bundle_file_sha256"], "terminal bundle file digest")
    _sha(terminal["terminal_bundle_digest"], "terminal bundle digest")
    _regular_sealed_file(
        terminal["terminal_bundle_path"],
        terminal["terminal_bundle_file_sha256"],
        "canonical terminal bundle",
        require_read_only=False,
    )
    try:
        terminal_bundle = json.loads(
            Path(terminal["terminal_bundle_path"]).read_text(encoding="ascii")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseCanonicalRowError(
            f"canonical terminal bundle cannot be reopened: {exc}"
        ) from exc
    if type(terminal_bundle) is not dict:
        raise WorkhorseCanonicalRowError("canonical terminal bundle is not an object")
    bundle_terminal = terminal_bundle.get("attempt_terminal")
    if (
        terminal_bundle.get("schema_version")
        != "crca_launchpad_attempt_terminal_bundle_v1"
        or terminal_bundle.get("attempt_id") != parent["attempt_id"]
        or canonical_sha256(terminal_bundle) != terminal["terminal_bundle_digest"]
        or type(bundle_terminal) is not dict
        or bundle_terminal.get("attempt_id") != parent["attempt_id"]
        or bundle_terminal.get("status") != terminal["attempt_terminal_status"]
        or bundle_terminal.get("final_frontiers") != terminal["final_frontiers"]
    ):
        raise WorkhorseCanonicalRowError(
            "canonical terminal projection differs from Scheduler terminal bundle"
        )
    frontiers = _exact_mapping(
        terminal["final_frontiers"],
        {
            "schema_version",
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        },
        "canonical terminal frontiers",
    )
    span = parent["absolute_tick_span"]
    start_tick = int(span["start_next_tick"])
    target = int(span["target_next_tick"])
    if frontiers["schema_version"] != "crca_attempt_frontiers_v1":
        raise WorkhorseCanonicalRowError("canonical terminal frontier schema changed")
    for field, frontier in frontiers.items():
        if field == "schema_version":
            continue
        value_int = _nonnegative_int(frontier, f"terminal {field}")
        if not start_tick <= value_int <= target:
            raise WorkhorseCanonicalRowError(
                "canonical terminal frontier leaves operation span"
            )
    core = {key: item for key, item in terminal.items() if key != "binding_sha256"}
    if terminal["binding_sha256"] != canonical_sha256(core):
        raise WorkhorseCanonicalRowError("canonical terminal binding digest changed")
    return terminal


def seal_workhorse_terminalized_device_envelope_calibration_receipt_v2(
    output_path: str | Path,
    *,
    calibration_receipt_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a calibration result only after exact Scheduler terminal reopen."""

    calibration = validate_workhorse_device_envelope_calibration_receipt_v2(
        calibration_receipt_path
    )
    parent = calibration["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        canonical_terminal_binding, parent=parent
    )
    target = parent["absolute_tick_span"]["target_next_tick"]
    if terminal["attempt_terminal_status"] != "COMPLETED" or any(
        terminal["final_frontiers"][field] != target
        for field in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        )
    ):
        raise WorkhorseCanonicalRowError(
            "calibration requires a completed exact-target terminal"
        )
    body = {
        "schema_version": WORKHORSE_TERMINALIZED_DEVICE_ENVELOPE_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "operation": "DEVICE_ENVELOPE_CALIBRATION",
        "shape_id": calibration["shape_id"],
        "backend_id": calibration["backend_id"],
        "compiled_temporal_k": sorted(
            int(value) for value in calibration["owner_components_by_k"]
        ),
        "artifact": calibration["artifact"],
        "device_identity": calibration["device_identity"],
        "capacity_basis": calibration["capacity_basis"],
        "calibration_binding": {
            "path": calibration["receipt_path"],
            "file_sha256": calibration["receipt_file_sha256"],
            "content_sha256": calibration["receipt_content_sha256"],
        },
        "canonical_parent_authority_binding": parent,
        "canonical_terminal_binding": terminal,
        "observation_mode": "OFF",
        "science_continuation_claimed": False,
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_terminalized_device_envelope_calibration_receipt_v2(
        output
    )


def validate_workhorse_terminalized_device_envelope_calibration_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen a terminalized calibration and recompute every binding."""

    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_TERMINALIZED_DEVICE_ENVELOPE_SCHEMA_V2,
        label="terminalized Workhorse device-envelope calibration receipt",
    )
    if (
        value.get("claim_class") != _CLAIM_CLASS
        or any(value.get(field) is not False for field in _NO_AUTHORITY)
        or value.get("operation") != "DEVICE_ENVELOPE_CALIBRATION"
        or value.get("observation_mode") != "OFF"
        or value.get("science_continuation_claimed") is not False
    ):
        raise WorkhorseCanonicalRowError(
            "terminalized calibration authority boundary changed"
        )
    binding = _exact_mapping(
        value.get("calibration_binding"),
        {"path", "file_sha256", "content_sha256"},
        "terminalized calibration binding",
    )
    calibration = validate_workhorse_device_envelope_calibration_receipt_v2(
        binding["path"]
    )
    parent = calibration["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        value.get("canonical_terminal_binding"), parent=parent
    )
    target = parent["absolute_tick_span"]["target_next_tick"]
    if (
        binding["file_sha256"] != calibration["receipt_file_sha256"]
        or binding["content_sha256"] != calibration["receipt_content_sha256"]
        or value.get("shape_id") != calibration["shape_id"]
        or value.get("backend_id") != calibration["backend_id"]
        or value.get("compiled_temporal_k")
        != sorted(int(item) for item in calibration["owner_components_by_k"])
        or value.get("artifact") != calibration["artifact"]
        or value.get("device_identity") != calibration["device_identity"]
        or value.get("capacity_basis") != calibration["capacity_basis"]
        or value.get("canonical_parent_authority_binding") != parent
        or terminal["attempt_terminal_status"] != "COMPLETED"
        or any(
            terminal["final_frontiers"][field] != target
            for field in (
                "last_dispatched_next_tick",
                "last_computed_next_tick",
                "durable_checkpoint_next_tick",
                "published_artifact_next_tick",
            )
        )
    ):
        raise WorkhorseCanonicalRowError(
            "terminalized calibration binding or terminal changed"
        )
    return value


def seal_workhorse_technical_row_receipt_v2(
    output_path: str | Path,
    *,
    execution_result_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the provider result only after canonical terminalization exists."""
    execution = validate_workhorse_canonical_row_execution_result_v2(
        execution_result_path
    )
    parent = execution["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        canonical_terminal_binding, parent=parent
    )
    outcome = execution["outcome"]
    target = execution["absolute_frontiers"]["target_next_tick"]
    if terminal["attempt_terminal_status"] != "COMPLETED" or any(
        terminal["final_frontiers"][field] != target
        for field in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        )
    ):
        raise WorkhorseCanonicalRowError(
            "executed row requires a completed exact-target terminal"
        )
    body = {
        "schema_version": WORKHORSE_TECHNICAL_ROW_RECEIPT_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "row_kind": "ATTEMPT_EXECUTED",
        "shape_id": execution["contract"]["shape_id"],
        "backend_id": execution["contract"]["backend_id"],
        "B": execution["contract"]["B"],
        "temporal_k": execution["contract"]["temporal_k"],
        "envelope_binding": execution["envelope_binding"],
        "artifact": execution["artifact"],
        "source_identity_sha256": execution["source_identity_sha256"],
        "device_identity": execution["device_identity"],
        "outcome": outcome,
        "p10_world_ticks_per_second": execution["p10_world_ticks_per_second"],
        "minimum_fixed_window_world_ticks_per_second": execution[
            "minimum_fixed_window_world_ticks_per_second"
        ],
        "median_fixed_window_world_ticks_per_second": execution[
            "median_fixed_window_world_ticks_per_second"
        ],
        "measurement_comparison_identity": (
            workhorse_fixed_tick_comparison_identity_v3(
                schedule=execution["measurement_schedule"],
                start_next_tick=execution["contract"]["start_next_tick"],
                source_checkpoint_identity_sha256=execution["checkpoint_binding"][
                    "source_checkpoint_identity_sha256"
                ],
            )
        ),
        "resource_ledger_conforms": execution["resource_observation"].get(
            "resource_ledger_conforms"
        ),
        "execution_result": {
            "path": execution["receipt_path"],
            "file_sha256": execution["receipt_file_sha256"],
            "content_sha256": execution["receipt_content_sha256"],
        },
        "canonical_parent_authority_binding": parent,
        "canonical_terminal_binding": terminal,
        "canonical_preflight_rejection_binding": None,
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_technical_row_receipt_v2(output)


def validate_workhorse_canonical_preflight_rejection_binding_v2(
    value: Mapping[str, Any],
    *,
    envelope: Mapping[str, Any],
    B: int,
    temporal_k: int,
) -> dict[str, Any]:
    binding = _exact_mapping(
        value,
        {
            "schema_version",
            "authority_source",
            "execution_authority_granted",
            "attempt_created",
            "reason_code",
            "preflight_receipt_path",
            "preflight_receipt_file_sha256",
            "execution_request_sha256",
            "campaign_plan_sha256",
            "envelope_receipt_content_sha256",
            "shape_id",
            "backend_id",
            "B",
            "temporal_k",
            "binding_sha256",
        },
        "canonical preflight rejection binding",
    )
    if (
        binding["schema_version"] != WORKHORSE_CANONICAL_PREFLIGHT_REJECTION_SCHEMA_V2
        or binding["authority_source"] != "CANONICAL_LAUNCHPAD_PREFLIGHT"
        or binding["execution_authority_granted"] is not False
        or binding["attempt_created"] is not False
        or binding["reason_code"] != "WORKHORSE_STABLE_CAPACITY_REJECTED"
        or binding["shape_id"] != envelope["shape_id"]
        or binding["backend_id"] != envelope["backend_id"]
        or binding["B"] != B
        or binding["temporal_k"] != temporal_k
        or binding["envelope_receipt_content_sha256"]
        != envelope["receipt_content_sha256"]
    ):
        raise WorkhorseCanonicalRowError("canonical preflight rejection changed row")
    for field in (
        "preflight_receipt_file_sha256",
        "execution_request_sha256",
        "campaign_plan_sha256",
        "envelope_receipt_content_sha256",
    ):
        _sha(binding[field], f"preflight {field}")
    _regular_sealed_file(
        binding["preflight_receipt_path"],
        binding["preflight_receipt_file_sha256"],
        "canonical preflight rejection receipt",
    )
    core = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if binding["binding_sha256"] != canonical_sha256(core):
        raise WorkhorseCanonicalRowError("preflight rejection binding digest changed")
    return binding


def seal_workhorse_preflight_rejection_row_receipt_v2(
    output_path: str | Path,
    *,
    envelope_receipt_path: str | Path,
    B: int,
    temporal_k: int,
    canonical_preflight_rejection_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one planned non-admission without fabricating an Attempt."""
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_receipt_path
    )
    row = _envelope_row(envelope, B=B, temporal_k=temporal_k)
    if row["planned_admission"].get("admitted") is not False:
        raise WorkhorseCanonicalRowError(
            "preflight rejection row was envelope-admitted"
        )
    preflight = validate_workhorse_canonical_preflight_rejection_binding_v2(
        canonical_preflight_rejection_binding,
        envelope=envelope,
        B=B,
        temporal_k=temporal_k,
    )
    body = {
        "schema_version": WORKHORSE_TECHNICAL_ROW_RECEIPT_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "row_kind": "PREFLIGHT_REJECTED_NO_ATTEMPT",
        "shape_id": envelope["shape_id"],
        "backend_id": envelope["backend_id"],
        "B": B,
        "temporal_k": temporal_k,
        "envelope_binding": {
            "receipt_path": envelope["receipt_path"],
            "receipt_file_sha256": envelope["receipt_file_sha256"],
            "receipt_content_sha256": envelope["receipt_content_sha256"],
        },
        "artifact": envelope["artifact"],
        "source_identity_sha256": envelope["source_identity_sha256"],
        "device_identity": envelope["device_identity"],
        "outcome": "PREFLIGHT_STABLE_CAPACITY_REJECTED",
        "p10_world_ticks_per_second": None,
        "minimum_fixed_window_world_ticks_per_second": None,
        "median_fixed_window_world_ticks_per_second": None,
        "measurement_comparison_identity": None,
        "resource_ledger_conforms": None,
        "execution_result": None,
        "canonical_parent_authority_binding": None,
        "canonical_terminal_binding": None,
        "canonical_preflight_rejection_binding": preflight,
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_technical_row_receipt_v2(output)


def validate_workhorse_technical_row_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_TECHNICAL_ROW_RECEIPT_SCHEMA_V2,
        label="Workhorse technical row receipt",
    )
    if value.get("claim_class") != _CLAIM_CLASS or any(
        value.get(field) is not False for field in _NO_AUTHORITY
    ):
        raise WorkhorseCanonicalRowError("technical row authority flags changed")
    envelope_binding = _exact_mapping(
        value.get("envelope_binding"),
        {"receipt_path", "receipt_file_sha256", "receipt_content_sha256"},
        "technical row envelope binding",
    )
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_binding["receipt_path"]
    )
    if (
        envelope_binding["receipt_file_sha256"] != envelope["receipt_file_sha256"]
        or envelope_binding["receipt_content_sha256"]
        != envelope["receipt_content_sha256"]
        or value.get("shape_id") != envelope["shape_id"]
        or value.get("backend_id") != envelope["backend_id"]
        or value.get("artifact") != envelope["artifact"]
        or value.get("source_identity_sha256") != envelope["source_identity_sha256"]
        or value.get("device_identity") != envelope["device_identity"]
    ):
        raise WorkhorseCanonicalRowError("technical row changed envelope identity")
    B = _positive_int(value.get("B"), "technical row B")
    temporal_k = _positive_int(value.get("temporal_k"), "technical row temporal_k")
    planned = _envelope_row(envelope, B=B, temporal_k=temporal_k)["planned_admission"]
    if value.get("row_kind") == "PREFLIGHT_REJECTED_NO_ATTEMPT":
        if (
            planned.get("admitted") is not False
            or value.get("outcome") != "PREFLIGHT_STABLE_CAPACITY_REJECTED"
            or value.get("p10_world_ticks_per_second") is not None
            or value.get("minimum_fixed_window_world_ticks_per_second") is not None
            or value.get("median_fixed_window_world_ticks_per_second") is not None
            or value.get("measurement_comparison_identity") is not None
            or value.get("resource_ledger_conforms") is not None
            or value.get("execution_result") is not None
            or value.get("canonical_parent_authority_binding") is not None
            or value.get("canonical_terminal_binding") is not None
        ):
            raise WorkhorseCanonicalRowError("preflight row fabricated execution")
        validate_workhorse_canonical_preflight_rejection_binding_v2(
            value.get("canonical_preflight_rejection_binding"),
            envelope=envelope,
            B=B,
            temporal_k=temporal_k,
        )
    elif value.get("row_kind") == "ATTEMPT_EXECUTED":
        if (
            planned.get("admitted") is not True
            or value.get("canonical_preflight_rejection_binding") is not None
        ):
            raise WorkhorseCanonicalRowError("executed row changed planned admission")
        result_binding = _exact_mapping(
            value.get("execution_result"),
            {"path", "file_sha256", "content_sha256"},
            "technical row execution result binding",
        )
        execution = validate_workhorse_canonical_row_execution_result_v2(
            result_binding["path"]
        )
        if (
            result_binding["file_sha256"] != execution["receipt_file_sha256"]
            or result_binding["content_sha256"] != execution["receipt_content_sha256"]
            or execution["contract"]["B"] != B
            or execution["contract"]["temporal_k"] != temporal_k
            or value.get("outcome") != execution["outcome"]
            or value.get("p10_world_ticks_per_second")
            != execution["p10_world_ticks_per_second"]
            or value.get("minimum_fixed_window_world_ticks_per_second")
            != execution["minimum_fixed_window_world_ticks_per_second"]
            or value.get("median_fixed_window_world_ticks_per_second")
            != execution["median_fixed_window_world_ticks_per_second"]
            or value.get("measurement_comparison_identity")
            != workhorse_fixed_tick_comparison_identity_v3(
                schedule=execution["measurement_schedule"],
                start_next_tick=execution["contract"]["start_next_tick"],
                source_checkpoint_identity_sha256=execution["checkpoint_binding"][
                    "source_checkpoint_identity_sha256"
                ],
            )
            or value.get("resource_ledger_conforms")
            != execution["resource_observation"].get("resource_ledger_conforms")
        ):
            raise WorkhorseCanonicalRowError("technical row execution result changed")
        parent = validate_workhorse_canonical_parent_authority_binding_v2(
            value.get("canonical_parent_authority_binding"),
            expected_operation="EXECUTION_ROW",
        )
        if parent != execution["canonical_parent_authority_binding"]:
            raise WorkhorseCanonicalRowError("technical row parent changed")
        terminal = validate_workhorse_canonical_terminal_binding_v2(
            value.get("canonical_terminal_binding"), parent=parent
        )
        target = execution["absolute_frontiers"]["target_next_tick"]
        if terminal["attempt_terminal_status"] != "COMPLETED" or any(
            terminal["final_frontiers"][field] != target
            for field in (
                "last_dispatched_next_tick",
                "last_computed_next_tick",
                "durable_checkpoint_next_tick",
                "published_artifact_next_tick",
            )
        ):
            raise WorkhorseCanonicalRowError(
                "persisted technical row requires exact completed terminal"
            )
    else:
        raise WorkhorseCanonicalRowError("technical row kind is unsupported")
    return value


def aggregate_workhorse_technical_rows_v2(
    output_path: str | Path,
    *,
    envelope_receipt_path: str | Path,
    row_receipt_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Offline complete-matrix validation and p10 selection; never authority."""
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_receipt_path
    )
    if type(row_receipt_paths) not in (list, tuple) or not row_receipt_paths:
        raise WorkhorseCanonicalRowError("aggregate row receipt paths are absent")
    rows = [
        validate_workhorse_technical_row_receipt_v2(path) for path in row_receipt_paths
    ]
    keys = [(row["temporal_k"], row["B"]) for row in rows]
    expected = [(row["temporal_k"], row["B"]) for row in envelope["expected_rows"]]
    if len(set(keys)) != len(keys):
        raise WorkhorseCanonicalRowError("aggregate contains a duplicate exact row")
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise WorkhorseCanonicalRowError("aggregate row coverage is incomplete")
    if any(
        row["envelope_binding"]["receipt_file_sha256"]
        != envelope["receipt_file_sha256"]
        or row["envelope_binding"]["receipt_content_sha256"]
        != envelope["receipt_content_sha256"]
        or row["artifact"] != envelope["artifact"]
        or row["source_identity_sha256"] != envelope["source_identity_sha256"]
        or row["device_identity"] != envelope["device_identity"]
        for row in rows
    ):
        raise WorkhorseCanonicalRowError(
            "aggregate mixes source, artifact, device, or envelope identity"
        )
    by_key = {(row["temporal_k"], row["B"]): row for row in rows}
    ordered = [by_key[key] for key in expected]
    candidates = [
        row
        for row in ordered
        if row["row_kind"] == "ATTEMPT_EXECUTED"
        and row["outcome"] == "MEASURED"
        and row["resource_ledger_conforms"] is True
        and type(row["p10_world_ticks_per_second"]) in (int, float)
        and type(row["minimum_fixed_window_world_ticks_per_second"]) in (int, float)
        and type(row["median_fixed_window_world_ticks_per_second"]) in (int, float)
        and type(row["measurement_comparison_identity"]) is dict
    ]
    comparison_identities = {
        canonical_sha256(row["measurement_comparison_identity"]) for row in candidates
    }
    if len(comparison_identities) > 1:
        raise WorkhorseCanonicalRowError(
            "aggregate mixes fixed-tick horizon, percentile, or checkpoint identity"
        )
    comparison_identity = (
        None if not candidates else candidates[0]["measurement_comparison_identity"]
    )
    winner = max(
        candidates,
        key=lambda row: (
            float(row["p10_world_ticks_per_second"]),
            float(row["minimum_fixed_window_world_ticks_per_second"]),
            float(row["median_fixed_window_world_ticks_per_second"]),
        ),
        default=None,
    )
    winner_projection = (
        None
        if winner is None
        else {
            "shape_id": winner["shape_id"],
            "backend_id": winner["backend_id"],
            "B": winner["B"],
            "temporal_k": winner["temporal_k"],
            "p10_world_ticks_per_second": winner["p10_world_ticks_per_second"],
            "minimum_fixed_window_world_ticks_per_second": winner[
                "minimum_fixed_window_world_ticks_per_second"
            ],
            "median_fixed_window_world_ticks_per_second": winner[
                "median_fixed_window_world_ticks_per_second"
            ],
            "measurement_comparison_identity_sha256": winner[
                "measurement_comparison_identity"
            ]["comparison_identity_sha256"],
            "row_receipt_content_sha256": winner["receipt_content_sha256"],
        }
    )
    body = {
        "schema_version": WORKHORSE_ROW_AGGREGATE_SCHEMA_V2,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "selection_authority": "OFFLINE_RECOMPUTATION_ONLY",
        "envelope_binding": {
            "receipt_path": envelope["receipt_path"],
            "receipt_file_sha256": envelope["receipt_file_sha256"],
            "receipt_content_sha256": envelope["receipt_content_sha256"],
        },
        "shape_id": envelope["shape_id"],
        "backend_id": envelope["backend_id"],
        "artifact": envelope["artifact"],
        "source_identity_sha256": envelope["source_identity_sha256"],
        "device_identity": envelope["device_identity"],
        "expected_row_count": len(expected),
        "retained_row_count": len(ordered),
        "measurement_comparison_identity": comparison_identity,
        "rows": [
            {
                "temporal_k": row["temporal_k"],
                "B": row["B"],
                "row_kind": row["row_kind"],
                "outcome": row["outcome"],
                "p10_world_ticks_per_second": row["p10_world_ticks_per_second"],
                "minimum_fixed_window_world_ticks_per_second": row[
                    "minimum_fixed_window_world_ticks_per_second"
                ],
                "median_fixed_window_world_ticks_per_second": row[
                    "median_fixed_window_world_ticks_per_second"
                ],
                "measurement_comparison_identity": row[
                    "measurement_comparison_identity"
                ],
                "receipt_path": row["receipt_path"],
                "receipt_file_sha256": row["receipt_file_sha256"],
                "receipt_content_sha256": row["receipt_content_sha256"],
            }
            for row in ordered
        ],
        "winner": winner_projection,
        "decision": (
            "TECHNICAL_SELECTION_AVAILABLE"
            if winner_projection is not None
            else "NO_CONFORMING_MEASURED_ROW"
        ),
        "future_science_use": (
            "A selected row must be concretized into a fresh canonical science "
            "ExecutionRequestV2; this aggregate is not a run or checkpoint identity."
        ),
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json(receipt, output_path)
    return validate_workhorse_technical_row_aggregate_v2(output)


def validate_workhorse_technical_row_aggregate_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_ROW_AGGREGATE_SCHEMA_V2,
        label="Workhorse row aggregate",
    )
    if (
        value.get("claim_class") != _CLAIM_CLASS
        or any(value.get(field) is not False for field in _NO_AUTHORITY)
        or value.get("selection_authority") != "OFFLINE_RECOMPUTATION_ONLY"
    ):
        raise WorkhorseCanonicalRowError("aggregate authority flags changed")
    binding = _exact_mapping(
        value.get("envelope_binding"),
        {"receipt_path", "receipt_file_sha256", "receipt_content_sha256"},
        "aggregate envelope binding",
    )
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        binding["receipt_path"]
    )
    rows = value.get("rows")
    if type(rows) is not list:
        raise WorkhorseCanonicalRowError("aggregate rows are malformed")
    row_paths = [row.get("receipt_path") for row in rows if type(row) is dict]
    if len(row_paths) != len(rows):
        raise WorkhorseCanonicalRowError("aggregate row locator is malformed")
    reopened = [validate_workhorse_technical_row_receipt_v2(path) for path in row_paths]
    expected_keys = [(row["temporal_k"], row["B"]) for row in envelope["expected_rows"]]
    observed_keys = [(row["temporal_k"], row["B"]) for row in reopened]
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        raise WorkhorseCanonicalRowError("aggregate row ordering/coverage changed")
    expected_rows = [
        {
            "temporal_k": row["temporal_k"],
            "B": row["B"],
            "row_kind": row["row_kind"],
            "outcome": row["outcome"],
            "p10_world_ticks_per_second": row["p10_world_ticks_per_second"],
            "minimum_fixed_window_world_ticks_per_second": row[
                "minimum_fixed_window_world_ticks_per_second"
            ],
            "median_fixed_window_world_ticks_per_second": row[
                "median_fixed_window_world_ticks_per_second"
            ],
            "measurement_comparison_identity": row["measurement_comparison_identity"],
            "receipt_path": row["receipt_path"],
            "receipt_file_sha256": row["receipt_file_sha256"],
            "receipt_content_sha256": row["receipt_content_sha256"],
        }
        for row in reopened
    ]
    candidates = [
        row
        for row in reopened
        if row["row_kind"] == "ATTEMPT_EXECUTED"
        and row["outcome"] == "MEASURED"
        and row["resource_ledger_conforms"] is True
    ]
    comparison_identities = {
        canonical_sha256(row["measurement_comparison_identity"]) for row in candidates
    }
    if len(comparison_identities) > 1:
        raise WorkhorseCanonicalRowError(
            "aggregate mixes fixed-tick horizon, percentile, or checkpoint identity"
        )
    expected_comparison_identity = (
        None if not candidates else candidates[0]["measurement_comparison_identity"]
    )
    winner = max(
        candidates,
        key=lambda row: (
            float(row["p10_world_ticks_per_second"]),
            float(row["minimum_fixed_window_world_ticks_per_second"]),
            float(row["median_fixed_window_world_ticks_per_second"]),
        ),
        default=None,
    )
    expected_winner = (
        None
        if winner is None
        else {
            "shape_id": winner["shape_id"],
            "backend_id": winner["backend_id"],
            "B": winner["B"],
            "temporal_k": winner["temporal_k"],
            "p10_world_ticks_per_second": winner["p10_world_ticks_per_second"],
            "minimum_fixed_window_world_ticks_per_second": winner[
                "minimum_fixed_window_world_ticks_per_second"
            ],
            "median_fixed_window_world_ticks_per_second": winner[
                "median_fixed_window_world_ticks_per_second"
            ],
            "measurement_comparison_identity_sha256": winner[
                "measurement_comparison_identity"
            ]["comparison_identity_sha256"],
            "row_receipt_content_sha256": winner["receipt_content_sha256"],
        }
    )
    if (
        binding["receipt_file_sha256"] != envelope["receipt_file_sha256"]
        or binding["receipt_content_sha256"] != envelope["receipt_content_sha256"]
        or value.get("shape_id") != envelope["shape_id"]
        or value.get("backend_id") != envelope["backend_id"]
        or value.get("artifact") != envelope["artifact"]
        or value.get("source_identity_sha256") != envelope["source_identity_sha256"]
        or value.get("device_identity") != envelope["device_identity"]
        or value.get("expected_row_count") != len(expected_keys)
        or value.get("retained_row_count") != len(expected_keys)
        or value.get("measurement_comparison_identity") != expected_comparison_identity
        or rows != expected_rows
        or value.get("winner") != expected_winner
        or value.get("decision")
        != (
            "TECHNICAL_SELECTION_AVAILABLE"
            if expected_winner is not None
            else "NO_CONFORMING_MEASURED_ROW"
        )
    ):
        raise WorkhorseCanonicalRowError("aggregate does not recompute")
    return value


__all__ = [
    "WORKHORSE_CANONICAL_PARENT_BINDING_SCHEMA_V2",
    "WORKHORSE_CANONICAL_PREFLIGHT_REJECTION_SCHEMA_V2",
    "WORKHORSE_CANONICAL_ROW_EXECUTION_SCHEMA_V2",
    "WORKHORSE_CANONICAL_TERMINAL_BINDING_SCHEMA_V2",
    "WORKHORSE_DEVICE_ENVELOPE_SCHEMA_V2",
    "WORKHORSE_FIXED_TICK_MEASUREMENT_SCHEDULE_SCHEMA_V3",
    "WORKHORSE_ROW_AGGREGATE_SCHEMA_V2",
    "WORKHORSE_TECHNICAL_ROW_RECEIPT_SCHEMA_V2",
    "WORKHORSE_TERMINALIZED_DEVICE_ENVELOPE_SCHEMA_V2",
    "WorkhorseCanonicalRowError",
    "aggregate_workhorse_technical_rows_v2",
    "run_and_seal_workhorse_device_envelope_calibration_v2",
    "run_workhorse_canonical_admitted_row_v2",
    "seal_workhorse_device_envelope_calibration_receipt_v2",
    "seal_workhorse_preflight_rejection_row_receipt_v2",
    "seal_workhorse_technical_row_receipt_v2",
    "seal_workhorse_terminalized_device_envelope_calibration_receipt_v2",
    "validate_workhorse_canonical_parent_authority_binding_v2",
    "validate_workhorse_canonical_preflight_rejection_binding_v2",
    "validate_workhorse_canonical_row_execution_result_v2",
    "validate_workhorse_canonical_terminal_binding_v2",
    "validate_workhorse_device_envelope_calibration_receipt_v2",
    "validate_workhorse_fixed_tick_measurement_schedule_v3",
    "validate_workhorse_live_device_under_semantic_parent_v2",
    "validate_workhorse_live_device_under_parent_v2",
    "validate_workhorse_technical_row_aggregate_v2",
    "validate_workhorse_technical_row_receipt_v2",
    "validate_workhorse_terminalized_device_envelope_calibration_receipt_v2",
    "workhorse_artifact_binding_v2",
    "workhorse_fixed_tick_comparison_identity_v3",
    "workhorse_fixed_tick_measurement_schedule_v3",
    "workhorse_row_checkpoint_binding_v2",
]
