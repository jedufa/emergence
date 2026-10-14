from __future__ import annotations

import copy
import hashlib
import inspect
import math
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .architecture_b_persistence_loop import ArchitectureBPersistenceLoop
from .bindings import load_prebuilt_extension
from .checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
    checkpoint_source_projection_authority_manifest,
    checkpoint_source_projection_digest_manifest,
)
from .compact_summary_digests import (
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
    active_agent_ids_digest_sha256,
    policy_mutation_tape_digest_from_tensor_sha256,
    policy_mutation_tape_digest_sha256,
)
from .interchange_projection import validate_native_checkpoint_projection
from .lifecycle_events import (
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
)
from .patch_profile import (
    PATCH_PROFILE_CLASSIFICATION_SCHEMA_VERSION,
    SUPPORTED_PATCH_PROFILES,
    classify_patch_architecture_b_profile,
)
from .runner_integration import validate_native_runtime_supported
from perf.cuda_architecture_b_checkpoint import (
    capture_phase_e_passive_modular_no_churn_guard,
    validate_phase_e_passive_modular_no_churn_guard,
)
from simulation.checkpoint_bundle_v2 import (
    validate_architecture_b_resume_envelope_authority,
)


SCHEMA_VERSION = "eiel_native_run_controller_facade_v1"
NATIVE_CONTROLLER_CLASS = "FlatCpuArchitectureBControllerV1"
NATIVE_CONTROLLER_VERSION = "eiel_native_architecture_b_controller_v1"
NATIVE_BOUNDARY_SCHEMA_VERSION = "eiel_native_architecture_b_boundary_v1"
NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION = (
    "eiel_native_policy_mutation_tape_boundary_authority_v1"
)
NATIVE_CHECKPOINT_PROJECTION_SCHEMA_VERSIONS = frozenset(
    {
        "cxxbackbone10_checkpoint_projection_v1",
        "cxxbackbone10_checkpoint_projection_v2",
    }
)
NATIVE_COMPACT_DIGEST_AUTHORITY_SCHEMA_VERSION = (
    "eiel_native_compact_digest_authority_v1"
)
NATIVE_RESULT_SCHEMA_VERSION = "eiel_native_architecture_b_run_result_v1"
NATIVE_CHECKPOINT_RESUME_ENVELOPE_SCHEMA_VERSION = (
    "eiel_architecture_b_checkpoint_resume_envelope_v1"
)
NATIVE_OBSERVER_CONTINUATION_SCHEMA_VERSION = (
    "eiel_architecture_b_observer_continuation_v1"
)
NATIVE_SNAPSHOT_SCHEDULE_SCHEMA_VERSION = "eiel_native_snapshot_schedule_v1"
NATIVE_CHECKPOINT_APPLY_SCHEMA_VERSION = "native_runtime_checkpoint_apply_v1"
NATIVE_CHECKPOINT_APPLY_SOURCE = (
    "checkpoint_bundle_v2.native_runtime_checkpoint_projection"
)
NATIVE_CONTROLLER_PROTOCOL_VERSION = 2
NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION = (
    "eiel_native_architecture_b_controller_protocol_v2"
)
NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE = (
    "EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_CAPABILITIES"
)
NATIVE_PATCH_CAPABILITIES_ATTRIBUTE = "EIEL_NATIVE_RESIDENT_PATCH_CAPABILITIES"
NATIVE_PATCH_CONTROLLED_COMPOSER_CAPABILITY = "controlled_architecture_b_cpu_v1"
NATIVE_INTER_TICK_CONTROL_CAPABILITY = "inter_tick_cancellation_control_v1"
NATIVE_PATCH_CONTROLLER_RECEIPT_SCHEMA_VERSION = (
    "eiel_native_architecture_b_patch_controller_receipt_v1"
)
NATIVE_PATCH_CONTROLLER_QUALIFICATION_SCHEMA_VERSION = (
    "eiel_native_architecture_b_patch_controller_qualification_v1"
)
NATIVE_PATCH_INSTRUMENTATION_RESULT_FIELD = (
    "architecture_b_patch_native_instrumentation"
)
NATIVE_PATCH_INSTRUMENTATION_SCHEMA_VERSION = (
    "eiel_native_architecture_b_patch_instrumentation_v1"
)
NATIVE_PATCH_INSTRUMENTATION_RECEIPT_SCHEMA_VERSION = (
    "eiel_architecture_b_patch_instrumentation_receipt_v1"
)
_PATCH_RESUME_BASE_CAPABILITY_MASK = {
    "flat": False,
    "patch": True,
    "churn": True,
}
_JUNGLE_NATIVE_EXECUTION_FLAGS = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)
NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY = "checkpoint_on_cancel_v1"
NATIVE_PHASE4_REQUIRED_CAPABILITIES = frozenset(
    {
        "acknowledgement_settlement_gate_v1",
        "complete_generation_cancellation_v1",
        "checkpoint_runtime_prefix_import_v1",
        NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY,
        "closed_snapshot_schedule_v1",
        "persistence_mailbox_v1",
        "resume_observer_continuation_v1",
        "two_phase_snapshot_validation_settlement_v1",
    }
)
NATIVE_SNAPSHOT_VALIDATION_PREPARE_RECEIPT_SCHEMA_VERSION = (
    "eiel_native_architecture_b_snapshot_validation_prepare_receipt_v1"
)
NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION = (
    "eiel_native_architecture_b_snapshot_validation_receipt_v1"
)
TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_REASON = (
    "tier4_juvenile_vulnerability_not_supported"
)
TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_MESSAGE = (
    "Architecture B does not support Tier 4 juvenile vulnerability"
)
_CANCEL_REASONS = frozenset(
    {
        "operator_requested",
        "signal_received",
        "supervisor_requested",
    }
)
_CANCEL_RECEIPT_FIELDS = frozenset(
    {
        "request_id",
        "controller_id",
        "owner_epoch",
        "disposition",
        "first_cancel_request_id",
        "cancel_sequence",
        "state_observed",
        "terminal_status_changed",
    }
)
_CANCEL_DISPOSITIONS = frozenset(
    {
        "accepted",
        "already_requested",
        "already_terminal",
        "not_running",
        "identity_mismatch",
        "control_point_unavailable",
    }
)
_TERMINAL_STATUSES = frozenset({"run_complete", "cancelled", "failed"})
_POLICY_MUTATION_TAPE_FIELDS = {
    "policy_birth_crossover_mask_tape": ("torch.bool", 2),
    "policy_birth_delta_tape": ("torch.float32", 2),
    "policy_birth_mutation_mask_tape": ("torch.bool", 2),
    "policy_birth_post_rng_state_tape": ("torch.uint8", 2),
    "policy_birth_semantic_draws_tape": ("torch.int64", 1),
}
_POLICY_MUTATION_TAPE_DENSE_POST_FIELD = "policy_birth_post_rng_state_tape"
_POLICY_MUTATION_TAPE_PACKED_POST_FIELD = "policy_birth_post_rng_state_packed_tape"
_POLICY_MUTATION_TAPE_PACKED_POST_SPEC = ("torch.uint8", 2)
_TORCH_CPU_RNG_FULL_STATE_BYTES_V1 = 5_056
_TORCH_CPU_RNG_PACKED_STATE_BYTES_V1 = 2_560
_POLICY_MUTATION_TAPE_CURSOR_FIELD = "policy_birth_tape_cursor"
_POLICY_MUTATION_TAPE_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "source_version",
        "cursor",
        "consumed_shapes",
        "tensor_sha256",
        "authority_complete",
        "export_non_consuming",
    }
)
_NATIVE_COMPACT_DIGEST_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "events_active_prefix_digest_sha256",
        "identity_active_agent_ids_digest_sha256",
        "rng_policy_mutation_tape_digest_sha256",
        "lifecycle_event_cursor",
        "active_agent_count",
        "policy_mutation_tape_cursor",
        "authority_complete_for_declared_fields",
        "declared_digest_count",
        "python_digest_execution",
        "full_checkpoint_projection_used",
    }
)
_SNAPSHOT_VALIDATION_PREPARE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "controller_id",
        "owner_epoch",
        "prepared",
        "disposition",
        "terminal_status",
        "trajectory_commit_frontier",
        "proposed_snapshot_frontier",
        "final_snapshot_sequence",
        "validated_source_projection_digest",
        "settlement_token",
        "prepared_receipt_digest_sha256",
        "success_marker_allowed",
    }
)
_SNAPSHOT_VALIDATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "controller_id",
        "owner_epoch",
        "acknowledged",
        "disposition",
        "terminal_status",
        "trajectory_commit_frontier",
        "snapshot_frontier",
        "final_snapshot_sequence",
        "validated_source_projection_digest",
        "success_marker_allowed",
    }
)
_SNAPSHOT_VALIDATION_SETTLEMENT_RECEIPT_FIELDS = (
    _SNAPSHOT_VALIDATION_RECEIPT_FIELDS
    | frozenset({"settlement_token", "prepared_receipt_digest_sha256"})
)
_PHASE4_RUN_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "decision",
        "run_id",
        "controller_id",
        "owner_epoch",
        "controller_protocol_version",
        "executed",
        "native_owner",
        "hidden_fallback",
        "runtime_jit",
        "terminal_status",
        "start_generation",
        "requested_generations",
        "completed_generations",
        "trajectory_commit_frontier",
        "snapshot_frontier",
        "next_generation",
        "next_tick",
        "controller_invocations",
        "python_generation_dispatches",
        "native_generation_transactions",
        "turnover_commits",
        "boundary_snapshot_count",
        "projection_export_count",
        "observer_live_owner_exports",
        "persistence_acknowledgement_count",
        "projection_tensor_bytes",
        "boundary_wall_time_ns",
        "boundaries",
        "final_snapshot_sequence",
        "success_marker_allowed",
        "promotion_result_allowed",
    }
)
_PHASE4_CANCELLED_RESULT_FIELDS = _PHASE4_RUN_RESULT_FIELDS | frozenset(
    {
        "cancel_request_id",
        "cancel_request_observation_cursor",
        "final_safe_cursor",
        "output_ack_frontier",
        "cancellation_checkpoint_requested",
        "cancellation_checkpoint_persisted",
        "cancelled_after_completed_generation",
        "cancelled_at_generation_boundary",
        "cancelled_snapshot_sequence",
        "cancelled_snapshot_id",
        "cancelled_snapshot_completion_disposition",
        "cancelled_snapshot_durable",
        "cancel_request_receipt",
        "cancel_observation_receipt",
        "cancel_terminal_receipt",
    }
)
_CANCEL_OBSERVATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "reason",
        "controller_id",
        "owner_epoch",
        "cancel_sequence",
        "observation_cursor",
        "observed_after_completed_generation",
        "boundary_ordinal",
        "snapshot_sequence",
        "snapshot_id",
        "snapshot_digest_sha256",
        "checkpoint_requested",
        "checkpoint_persisted",
    }
)
_CANCEL_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "controller_id",
        "owner_epoch",
        "terminal_status",
        "request_observation_cursor",
        "final_safe_cursor",
        "trajectory_commit_frontier",
        "snapshot_frontier",
        "output_ack_frontier",
        "snapshot_sequence",
        "snapshot_id",
        "snapshot_digest_sha256",
        "cancellation_checkpoint_requested",
        "cancellation_checkpoint_persisted",
        "success_marker_allowed",
        "promotion_result_allowed",
    }
)
_PHASE4_FAILED_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "decision",
        "run_id",
        "controller_id",
        "owner_epoch",
        "executed",
        "native_owner",
        "hidden_fallback",
        "runtime_jit",
        "terminal_status",
        "start_generation",
        "requested_generations",
        "completed_generations",
        "trajectory_commit_frontier",
        "snapshot_frontier",
        "active_generation",
        "controller_invocations",
        "trajectory_dispatch_count",
        "python_generation_dispatches",
        "boundaries",
        "root_failure",
        "success_marker_allowed",
    }
)
_PHASE4_ROOT_FAILURE_FIELDS = frozenset(
    {
        "error_sequence",
        "error_code",
        "origin",
        "run_id",
        "controller_id",
        "owner_epoch",
        "active_generation",
        "message",
        "owner_integrity",
        "retry_disposition",
    }
)
_PERSISTENCE_LOOP_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "stop_reason",
        "terminal_state_seen",
        "snapshots_seen",
        "progress_only_snapshots",
        "skipped_snapshot_sequences",
        "persistence_acknowledgements",
        "identical_acknowledgement_retries",
        "checkpoints_persisted",
        "architecture_b_checkpoint_bundle_paths",
        "last_snapshot_sequence",
        "last_progress",
    }
)
_RESUME_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_kind",
        "source_projection_sha256",
        "checkpoint_projection",
        "architecture_b_identity",
        "observer_continuation",
        "controller_import_metadata",
        "bundle_completion_authority",
    }
)
_ARCHITECTURE_B_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "controller_id",
        "owner_epoch",
        "snapshot_id",
        "snapshot_sequence",
        "snapshot_digest_sha256",
        "next_generation",
        "tick",
        "phase",
        "source_projection_sha256",
    }
)
_OBSERVER_CONTINUATION_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_next_generation",
        "tick",
        "typed_lifecycle_event_cursor",
        "exact_reporting_boundary_ordinal",
        "behavior_completed_generation_and_row_key",
        "metrics_completed_generation",
        "lineage_completed_generation_and_row_key",
        "chronicle_observation_ordinal",
        "progress_sequence",
        "schedule_phase",
        "deduplication",
        "acknowledged_channel_cursor_ranges",
    }
)
_ACKNOWLEDGED_CHANNELS = frozenset(
    {
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
        "chronicle",
        "checkpoint_generation",
    }
)
_SNAPSHOT_SCHEDULE_FIELDS = frozenset(
    {
        "schema_version",
        "progress_every_generations",
        "report_every_generations",
        "checkpoint_every_generations",
        "checkpoint_on_cancel",
        "explicit_next_generations",
        "emit_final_snapshot",
        "typed_lifecycle_mode",
        "mailbox_capacity",
    }
)
_PATCH_NATIVE_INSTRUMENTATION_FIELDS = frozenset(
    {
        "host_serialization_bytes",
        "host_projection_time_ns",
        "device_transfer_time_ns",
        "device_synchronization_time_ns",
        "allocation_count",
        "allocation_bytes",
        "simulation_time_ns",
        "observation_time_ns",
        "policy_time_ns",
        "world_patch_time_ns",
        "lifecycle_time_ns",
        "boundary_settlement_time_ns",
        "checkpoint_projection_time_ns",
        "reporting_projection_time_ns",
    }
)
_PATCH_INSTRUMENTATION_MEASUREMENT_FIELDS = (
    "initial_import_bytes",
    "generation_boundary_projection_bytes",
    "checkpoint_bytes",
    "host_serialization_bytes",
    "host_projection_time_ns",
    "device_transfer_time_ns",
    "device_synchronization_time_ns",
    "allocation_count",
    "allocation_bytes",
    "controller_boundary_wall_time_ns",
    "simulation_time_ns",
    "observation_time_ns",
    "policy_time_ns",
    "world_patch_time_ns",
    "lifecycle_time_ns",
    "boundary_settlement_time_ns",
    "checkpoint_projection_time_ns",
    "reporting_projection_time_ns",
)
_PATCH_INSTRUMENTATION_UNITS = {
    field: (
        "count"
        if field == "allocation_count"
        else "nanoseconds"
        if field.endswith("_time_ns")
        else "bytes"
    )
    for field in _PATCH_INSTRUMENTATION_MEASUREMENT_FIELDS
}
_MISSING = object()


class NativeRunControllerError(RuntimeError):
    """Raised when the Architecture B native run cannot be trusted."""


def _fail(message: str) -> NativeRunControllerError:
    return NativeRunControllerError(f"EIEL-NATIVE-RUN-CONTROLLER-FAILED: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_run_id(runner: Any) -> str:
    sim_cfg = getattr(runner, "sim_cfg", None)
    configured = str(getattr(sim_cfg, "chronicle_run_id", "") or "").strip()
    if configured:
        return configured
    run_dir_name = Path(str(getattr(runner, "run_dir", "") or "")).name.strip()
    return _strict_identifier(run_dir_name, field="runner run_dir name")


def _detached_copy(value: Any) -> Any:
    object_memo: dict[int, Any] = {}
    deepcopy_memo: dict[int, Any] = {}
    detached_tensors: list[Any] = []

    def clone(item: Any) -> Any:
        item_id = id(item)
        if item_id in object_memo:
            return object_memo[item_id]

        if isinstance(item, Mapping):
            copied_mapping: dict[Any, Any] = {}
            object_memo[item_id] = copied_mapping
            for key, nested in item.items():
                copied_mapping[clone(key)] = clone(nested)
            return copied_mapping
        if isinstance(item, list):
            copied_list: list[Any] = []
            object_memo[item_id] = copied_list
            copied_list.extend(clone(nested) for nested in item)
            return copied_list
        if isinstance(item, tuple):
            copied_tuple = tuple(clone(nested) for nested in item)
            object_memo[item_id] = copied_tuple
            return copied_tuple
        if hasattr(item, "detach") and hasattr(item, "clone"):
            detached = item.detach()
            detached_tensors.append(detached)
            copied_tensor = copy.deepcopy(detached, deepcopy_memo)
            object_memo[item_id] = copied_tensor
            return copied_tensor

        copied_item = copy.deepcopy(item, deepcopy_memo)
        object_memo[item_id] = copied_item
        return copied_item

    return clone(value)


def _dict_copy(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} is not a mapping")
    copied = _detached_copy(value)
    if not isinstance(copied, dict):
        raise _fail(f"{field} did not normalize to a dictionary")
    return copied


def _strict_int(value: Any, *, field: str, minimum: int) -> int:
    if type(value) is not int:
        raise _fail(f"{field} must be an integer")
    resolved = value
    if resolved < minimum:
        raise _fail(f"{field} must be at least {minimum}")
    return resolved


def _strict_identifier(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise _fail(f"{field} must be a non-empty identifier")
    return value


def _strict_ascii_identifier(value: Any, *, field: str) -> str:
    resolved = _strict_identifier(value, field=field)
    try:
        encoded = resolved.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _fail(f"{field} must contain only ASCII characters") from exc
    if len(encoded) > 128:
        raise _fail(f"{field} must be at most 128 bytes")
    return resolved


def _strict_lowercase_sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise _fail(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def _explicit_patch_extension_artifact(
    extension_path: str | Path | None,
    expected_extension_sha256: Any,
) -> tuple[Path, str]:
    if extension_path is None:
        raise _fail(
            "Patch Architecture B controller requires an explicit development extension"
        )
    selected_path = Path(extension_path).expanduser().resolve()
    if not selected_path.is_file():
        raise _fail(f"Architecture B extension does not exist: {selected_path}")
    expected_sha256 = _strict_lowercase_sha256(
        expected_extension_sha256,
        field="explicit Architecture B extension SHA-256",
    )
    actual_sha256 = _sha256(selected_path)
    if actual_sha256 != expected_sha256:
        raise _fail(
            "explicit Architecture B extension SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return selected_path, actual_sha256


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise _fail(f"{field} must be {expected}")


def _closed_dict(
    value: Any,
    *,
    field: str,
    required_fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise _fail(f"{field} must be an object")
    actual = frozenset(value)
    missing = sorted(required_fields - actual)
    unexpected = sorted(actual - required_fields)
    if missing or unexpected:
        raise _fail(
            f"{field} fields mismatch: missing={missing} unexpected={unexpected}"
        )
    return value


def _exact_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_exact_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _exact_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if hasattr(left, "detach") or hasattr(right, "detach"):
        if not hasattr(left, "detach") or not hasattr(right, "detach"):
            return False
        left_tensor = left.detach().to("cpu")
        right_tensor = right.detach().to("cpu")
        if left_tensor.dtype != right_tensor.dtype or tuple(left_tensor.shape) != tuple(
            right_tensor.shape
        ):
            return False
        if left_tensor.equal(right_tensor):
            return True
        if left_tensor.is_complex():
            return _exact_values_equal(
                left_tensor.real, right_tensor.real
            ) and _exact_values_equal(left_tensor.imag, right_tensor.imag)
        if not left_tensor.is_floating_point():
            return False
        left_nan = left_tensor.isnan()
        right_nan = right_tensor.isnan()
        return bool(
            left_nan.equal(right_nan)
            and left_tensor[~left_nan].equal(right_tensor[~right_nan])
        )
    try:
        compared = left == right
        return bool(compared if isinstance(compared, bool) else compared.all())
    except (AttributeError, TypeError, ValueError):
        return False


def _configured_world(runner: Any) -> str:
    env_cfg = getattr(runner, "env_cfg", None)
    configured = getattr(env_cfg, "world", None)
    if configured is None or not str(configured).strip():
        configured = getattr(env_cfg, "world_mode", "flat")
    return str(configured or "flat").strip().lower()


def _classify_architecture_b_world_profile(
    runner: Any,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Resolve Flat unchanged or validate the closed Patch controller envelope."""

    env_cfg = getattr(runner, "env_cfg", None)
    if env_cfg is None:
        return "flat", None, None

    explicit_world = getattr(env_cfg, "world", _MISSING)
    if explicit_world is _MISSING:
        explicit_world = getattr(env_cfg, "world_mode", _MISSING)
    patch_gate = getattr(env_cfg, "patch_world_enable", _MISSING)

    if explicit_world is _MISSING:
        if patch_gate is _MISSING or patch_gate is False:
            return "flat", None, None
        classification_world: Any | None = None
    elif type(explicit_world) is str and explicit_world.strip().lower() == "flat":
        if patch_gate is _MISSING or patch_gate is False:
            return "flat", None, None
        classification_world = explicit_world
    else:
        classification_world = explicit_world

    wdl_enabled = getattr(env_cfg, "wdl_delayed_hazard_enable", False)
    wdl_observation_enabled = getattr(
        env_cfg,
        "wdl_delayed_hazard_observation_enable",
        False,
    )
    if type(wdl_enabled) is not bool or type(wdl_observation_enabled) is not bool:
        raise _fail("Patch WDL feature switches must be boolean")
    if wdl_observation_enabled and not wdl_enabled:
        raise _fail("Patch WDL observations require active WDL execution")
    jungle_flags: dict[str, bool] = {}
    for name in _JUNGLE_NATIVE_EXECUTION_FLAGS:
        value = getattr(env_cfg, name, False)
        if type(value) is not bool:
            raise _fail(f"Patch Jungle feature switch {name} must be boolean")
        jungle_flags[name] = value
    jungle_enabled = any(jungle_flags.values())
    classification_env_cfg = env_cfg
    if wdl_enabled or jungle_enabled:
        classification_env_cfg = copy.copy(env_cfg)
        if wdl_enabled:
            classification_env_cfg.wdl_delayed_hazard_enable = False
            classification_env_cfg.wdl_delayed_hazard_observation_enable = False
        if jungle_enabled:
            for name in _JUNGLE_NATIVE_EXECUTION_FLAGS:
                setattr(classification_env_cfg, name, False)

    classification = classify_patch_architecture_b_profile(
        getattr(runner, "sim_cfg", None),
        classification_env_cfg,
        runner,
        world=classification_world,
    )
    if (
        classification.schema_version
        != PATCH_PROFILE_CLASSIFICATION_SCHEMA_VERSION
        or type(classification.supported) is not bool
        or not isinstance(classification.unsupported_reasons, tuple)
        or any(
            type(reason) is not str or not reason
            for reason in classification.unsupported_reasons
        )
    ):
        raise _fail("Patch profile classifier returned a malformed decision")
    if not classification.supported:
        reasons = ",".join(classification.unsupported_reasons) or (
            "unsupported_patch_profile"
        )
        raise _fail(f"Patch Architecture B profile is unsupported: {reasons}")
    if classification.profile not in SUPPORTED_PATCH_PROFILES:
        raise _fail("Patch profile classifier returned an unsupported profile")
    classification_payload = classification.to_dict()
    classification_payload["wdl_native_execution"] = wdl_enabled
    classification_payload["wdl_native_observation"] = wdl_observation_enabled
    classification_payload["jungle_native_execution"] = jungle_enabled
    classification_payload["jungle_native_features"] = copy.deepcopy(jungle_flags)
    return "patch", classification.profile, classification_payload


def _validate_patch_import_projection(
    metadata: Any,
    *,
    expected_profile: str,
) -> None:
    if not isinstance(metadata, Mapping):
        raise _fail("Patch Architecture B import metadata is not a mapping")
    if metadata.get("world") != "patch":
        raise _fail("Patch Architecture B import did not declare world=patch")
    if metadata.get("native_runtime_patch_profile") != expected_profile:
        raise _fail("Patch Architecture B import profile contradicts classification")
    if metadata.get("native_runtime_patch_contract_schema_version") != (
        "eiel_patch_native_import_contract_v1"
    ):
        raise _fail("Patch Architecture B import contract schema mismatch")
    if metadata.get("device") != "cpu" or metadata.get("bank_device") != "cpu":
        raise _fail("Patch Architecture B import requires CPU devices")
    if metadata.get("native_runtime_allow_runtime_jit_build") is not False:
        raise _fail("Patch Architecture B import permits runtime JIT")
    capabilities = metadata.get("capability_mask")
    if not isinstance(capabilities, Mapping):
        raise _fail("Patch Architecture B import has no capability mask")
    if (
        capabilities.get("patch") is not True
        or capabilities.get("flat") is not False
        or capabilities.get("churn") is not True
    ):
        raise _fail("Patch Architecture B import capability mask is inconsistent")
    for name in ("gru", "cuda"):
        if bool(capabilities.get(name, False)):
            raise _fail(
                f"Patch Architecture B import capability {name} is enabled"
            )
    wdl_execution = metadata.get("native_runtime_wdl_execution_enable", False)
    if type(wdl_execution) is not bool:
        raise _fail("Patch Architecture B import WDL execution marker is malformed")
    if capabilities.get("wdl", False) is not wdl_execution:
        raise _fail("Patch Architecture B import WDL capability is inconsistent")
    jungle_execution = metadata.get("native_runtime_jungle_execution_enable", False)
    if type(jungle_execution) is not bool:
        raise _fail("Patch Architecture B import Jungle execution marker is malformed")
    if capabilities.get("jungle", False) is not jungle_execution:
        raise _fail("Patch Architecture B import Jungle capability is inconsistent")


def _patch_resume_capability_mask(metadata: Mapping[str, Any]) -> dict[str, bool]:
    mask = dict(_PATCH_RESUME_BASE_CAPABILITY_MASK)
    if metadata.get("native_runtime_wdl_execution_enable", False) is True:
        mask["wdl"] = True
    if metadata.get("native_runtime_jungle_execution_enable", False) is True:
        mask["jungle"] = True
    return mask


def _patch_extension_capabilities(module: Any) -> dict[str, bool]:
    raw = getattr(module, NATIVE_PATCH_CAPABILITIES_ATTRIBUTE, None)
    if type(raw) is not dict:
        raise _fail("prebuilt extension has no controlled Patch capability receipt")
    expected = {
        NATIVE_PATCH_CONTROLLED_COMPOSER_CAPABILITY: True,
        "resident_action_cache_cpu_v1": True,
        "resident_population_birth_commit_cpu_v1": True,
        "promotion_accepted": False,
        "gru_supported": False,
        "cuda_supported": False,
    }
    optional_family_fields = ("jungle_supported", "wdl_supported")
    expected_fields = frozenset(
        {*expected, *optional_family_fields, "exact_parity_qualified"}
    )
    if frozenset(raw) != expected_fields:
        raise _fail(
            "prebuilt extension Patch capability fields mismatch: "
            f"missing={sorted(expected_fields - frozenset(raw))} "
            f"unexpected={sorted(frozenset(raw) - expected_fields)}"
        )
    normalized: dict[str, bool] = {}
    for name, expected_value in expected.items():
        value = raw.get(name)
        if type(value) is not bool or value is not expected_value:
            raise _fail(
                "prebuilt extension Patch capability mismatch: "
                f"{name} must be {expected_value}"
            )
        normalized[name] = value
    for name in optional_family_fields:
        value = raw.get(name)
        if type(value) is not bool:
            raise _fail(
                f"prebuilt extension Patch capability {name} must be boolean"
            )
        normalized[name] = value
    exact_parity = raw.get("exact_parity_qualified")
    if type(exact_parity) is not bool:
        raise _fail(
            "prebuilt extension Patch capability exact_parity_qualified must be boolean"
        )
    normalized["exact_parity_qualified"] = exact_parity
    return normalized


def _validate_runner_projection_configuration(
    runner: Any,
    metadata: Mapping[str, Any],
) -> tuple[int, int]:
    sim_cfg = getattr(runner, "sim_cfg", None)
    configured_steps = _strict_int(
        getattr(sim_cfg, "steps_per_generation", None),
        field="runner steps_per_generation",
        minimum=1,
    )
    projection_steps = _strict_int(
        metadata.get("configured_steps_per_generation"),
        field="checkpoint configured_steps_per_generation",
        minimum=1,
    )
    if projection_steps != configured_steps:
        raise _fail(
            "checkpoint steps_per_generation does not match the qualified runner"
        )

    configured_world = _configured_world(runner)
    if configured_world != "flat" or metadata.get("world") != configured_world:
        raise _fail("checkpoint world does not exactly match the Flat runner configuration")

    configured_device = str(getattr(sim_cfg, "device", "") or "").strip().lower()
    if configured_device != "cpu" or metadata.get("device") != configured_device:
        raise _fail("checkpoint device does not exactly match the CPU runner configuration")

    configured_policy = str(
        getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or ""
    ).strip().lower()
    if configured_policy != "mlp" or metadata.get(
        "native_runtime_fixed_policy_family"
    ) != configured_policy:
        raise _fail("checkpoint policy does not exactly match the fixed-MLP configuration")

    max_births = _strict_int(
        getattr(sim_cfg, "native_runtime_max_births_per_generation", 1),
        field="native_runtime_max_births_per_generation",
        minimum=0,
    )
    return configured_steps, max_births


def _phase_e_passive_no_churn_guard_for_controller(
    sim_cfg: Any,
    projection: Mapping[str, Any],
    *,
    max_births: int,
) -> dict[str, Any] | None:
    enabled = getattr(sim_cfg, "native_runtime_modular_storage_enable", False)
    if type(enabled) is not bool:
        raise _fail("native_runtime_modular_storage_enable must be an exact bool")
    if enabled is False:
        return None
    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("Phase E passive modular checkpoint metadata is malformed")
    capacity = metadata.get("capacity")
    if type(capacity) is not int or capacity <= 0:
        raise _fail("Phase E passive modular checkpoint capacity is invalid")
    try:
        return capture_phase_e_passive_modular_no_churn_guard(
            projection,
            enabled=True,
            fixed_capacity=capacity,
            max_births=max_births,
        )
    except Exception as exc:
        raise _fail(f"Phase E passive modular no-churn import rejected: {exc}") from exc


def _validate_patch_resume_projection_configuration(
    runner: Any,
    metadata: Mapping[str, Any],
    *,
    expected_profile: str,
) -> tuple[int, int]:
    _validate_patch_import_projection(
        metadata,
        expected_profile=expected_profile,
    )
    capabilities = metadata.get("capability_mask")
    expected_capabilities = _patch_resume_capability_mask(metadata)
    if type(capabilities) is not dict or capabilities != expected_capabilities:
        raise _fail(
            "Patch checkpoint capability metadata must exactly match the finite CPU envelope"
        )

    sim_cfg = getattr(runner, "sim_cfg", None)
    configured_steps = _strict_int(
        getattr(sim_cfg, "steps_per_generation", None),
        field="runner steps_per_generation",
        minimum=1,
    )
    projection_steps = _strict_int(
        metadata.get("configured_steps_per_generation"),
        field="checkpoint configured_steps_per_generation",
        minimum=1,
    )
    if projection_steps != configured_steps:
        raise _fail(
            "checkpoint steps_per_generation does not match the qualified Patch runner"
        )
    if getattr(sim_cfg, "device", None) != "cpu":
        raise _fail("Patch checkpoint runner device must exactly equal cpu")
    if getattr(sim_cfg, "native_runtime_fixed_policy_family", None) != "mlp":
        raise _fail("Patch checkpoint runner policy must exactly equal fixed MLP")

    max_births = _strict_int(
        getattr(sim_cfg, "native_runtime_max_births_per_generation", 1),
        field="native_runtime_max_births_per_generation",
        minimum=0,
    )
    return configured_steps, max_births


def _validate_patch_restored_engine(
    engine: Any,
    *,
    expected_profile: str,
    expected_capability_mask: Mapping[str, bool],
) -> None:
    if (
        getattr(engine, "world", None) != "patch"
        or getattr(engine, "patch_profile", None) != expected_profile
        or type(getattr(engine, "capability_mask", None)) is not dict
        or getattr(engine, "capability_mask", None) != dict(expected_capability_mask)
    ):
        raise _fail("Patch checkpoint restore engine authority metadata mismatch")
    capability = getattr(engine, "capability", None)
    if (
        capability is None
        or getattr(capability, "accepted_default_native_envelope", None) is not False
        or getattr(capability, "supports_patch_generation", None) is not False
        or getattr(capability, "supports_cxx19_formal_benchmark_gate", None) is not False
        or getattr(capability, "runtime_jit_in_backend", None) is not False
        or getattr(capability, "cuda", None) is not False
        or getattr(capability, "benchmark_qualification", None) != "not_qualified"
        or hasattr(engine, "safe_default_decision")
    ):
        raise _fail("Patch checkpoint restore engine exposed a promoted capability")


def _restore_patch_checkpoint_projection(
    projection: Mapping[str, Any],
    *,
    extension_path: Path,
    expected_profile: str,
    source_projection_sha256: str,
) -> dict[str, Any]:
    from .engine import NativeEielEngine

    try:
        restored_engine = NativeEielEngine.from_checkpoint_projection(
            _dict_copy(projection, field="Patch checkpoint restore projection"),
            extension_path=extension_path,
        )
    except Exception as exc:
        raise _fail(f"Patch checkpoint restore engine rejected the projection: {exc}") from exc

    _validate_patch_restored_engine(
        restored_engine,
        expected_profile=expected_profile,
        expected_capability_mask=_patch_resume_capability_mask(
            projection["metadata"]
        ),
    )
    counters = getattr(restored_engine, "counters", None)
    to_dict = getattr(counters, "to_dict", None)
    counter_values = to_dict() if callable(to_dict) else None
    if (
        not isinstance(counter_values, Mapping)
        or counter_values.get("engine_constructions") != 1
        or counter_values.get("checkpoint_restore_calls") != 1
        or counter_values.get("import_projection_calls") != 0
        or counter_values.get("generation_calls") != 0
    ):
        raise _fail("Patch checkpoint restore engine provenance counters are invalid")

    export_projection = getattr(restored_engine, "export_checkpoint_projection", None)
    if not callable(export_projection):
        raise _fail("Patch checkpoint restore engine cannot export restored authority")
    try:
        restored = _dict_copy(
            export_projection(),
            field="restored Patch checkpoint projection",
        )
        validate_native_checkpoint_projection(restored)
    except Exception as exc:
        raise _fail(
            f"restored Patch checkpoint projection validation failed: {exc}"
        ) from exc
    source_restore_count = _strict_int(
        projection.get("restore_count"),
        field="source Patch checkpoint restore_count",
        minimum=0,
    )
    restored_restore_count = _strict_int(
        restored.get("restore_count"),
        field="restored Patch checkpoint restore_count",
        minimum=1,
    )
    if restored_restore_count != source_restore_count + 1:
        raise _fail("restored Patch checkpoint restore_count is not authoritative")
    restored["restore_count"] = source_restore_count
    try:
        restored_authority = checkpoint_source_projection_authority_manifest(restored)
    except Exception as exc:
        raise _fail(
            f"normalized Patch checkpoint authority validation failed: {exc}"
        ) from exc
    if restored_authority.get("source_projection_sha256") != source_projection_sha256:
        raise _fail("restored Patch checkpoint source authority digest mismatch")

    restored_metadata = restored.get("metadata")
    if not isinstance(restored_metadata, Mapping):
        raise _fail("restored Patch checkpoint metadata is malformed")
    _validate_patch_import_projection(
        restored_metadata,
        expected_profile=expected_profile,
    )
    if restored_metadata.get("capability_mask") != _patch_resume_capability_mask(
        restored_metadata
    ):
        raise _fail("restored Patch checkpoint capability metadata mismatch")
    return restored


def _restored_native_checkpoint_projection(
    runner: Any,
    *,
    generation: int,
    tick: int,
    expected_patch_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    raw_apply = getattr(runner, "_native_runtime_checkpoint_apply", None)
    if raw_apply is None:
        return None
    apply_summary = _dict_copy(
        raw_apply,
        field="runner native checkpoint apply summary",
    )
    if apply_summary.get("schema_version") != NATIVE_CHECKPOINT_APPLY_SCHEMA_VERSION:
        raise _fail("runner native checkpoint apply summary schema mismatch")
    if apply_summary.get("status") != "restored":
        raise _fail("runner native checkpoint apply is not restored")
    if apply_summary.get("source") != NATIVE_CHECKPOINT_APPLY_SOURCE:
        raise _fail("runner native checkpoint apply source is not authoritative")
    if apply_summary.get("verification_target") != "restored_native_engine":
        raise _fail("runner native checkpoint apply did not verify a restored native owner")
    if "destination_runtime" in apply_summary:
        raise _fail("runner native checkpoint apply materialized a non-native destination")
    source_projection_sha256 = _strict_lowercase_sha256(
        apply_summary.get("source_projection_sha256"),
        field="runner native checkpoint source projection SHA-256",
    )

    applied_generation = _strict_int(
        apply_summary.get("generation"),
        field="runner native checkpoint apply generation",
        minimum=0,
    )
    applied_tick = _strict_int(
        apply_summary.get("tick"),
        field="runner native checkpoint apply tick",
        minimum=0,
    )
    if applied_generation != generation or applied_tick != tick:
        raise _fail("runner native checkpoint apply cursor contradicts the requested import")

    from .engine import NativeEielEngine

    engine = getattr(runner, "_native_eiel_engine", None)
    if not isinstance(engine, NativeEielEngine) or getattr(engine, "runner", None) is not runner:
        raise _fail("runner native checkpoint apply has no bound restored native owner")
    if expected_patch_profile is not None:
        _validate_patch_restored_engine(
            engine,
            expected_profile=expected_patch_profile,
        )
    counters = getattr(engine, "counters", None)
    to_dict = getattr(counters, "to_dict", None)
    if not callable(to_dict):
        raise _fail("restored native owner has no checkpoint provenance counters")
    counter_values = to_dict()
    if not isinstance(counter_values, Mapping):
        raise _fail("restored native owner checkpoint provenance counters are malformed")
    if (
        counter_values.get("checkpoint_restore_calls") != 1
        or counter_values.get("import_projection_calls") != 0
        or counter_values.get("generation_calls") != 0
    ):
        raise _fail("restored native owner checkpoint provenance counters are invalid")

    export_projection = getattr(engine, "export_checkpoint_projection", None)
    if not callable(export_projection):
        raise _fail("restored native owner cannot export its checkpoint projection")
    try:
        restored_projection = _dict_copy(
            export_projection(),
            field="restored native owner checkpoint projection",
        )
        validate_native_checkpoint_projection(restored_projection)
        restored_authority = checkpoint_source_projection_authority_manifest(
            restored_projection
        )
    except Exception as exc:
        raise _fail(
            f"restored native owner checkpoint projection validation failed: {exc}"
        ) from exc

    raw_source_projection = getattr(
        runner,
        "_native_runtime_checkpoint_source_projection",
        None,
    )
    if raw_source_projection is None:
        raise _fail("restored native owner has no bound source checkpoint projection")
    try:
        projection = _dict_copy(
            raw_source_projection,
            field="bound source checkpoint projection",
        )
        validate_native_checkpoint_projection(projection)
        source_authority = checkpoint_source_projection_authority_manifest(projection)
    except Exception as exc:
        raise _fail(
            f"bound source checkpoint projection validation failed: {exc}"
        ) from exc
    if source_authority.get("source_projection_sha256") != source_projection_sha256:
        raise _fail("bound source checkpoint projection digest contradicts apply receipt")

    if expected_patch_profile is not None:
        source_restore_count = _strict_int(
            projection.get("restore_count"),
            field="bound source Patch checkpoint restore_count",
            minimum=0,
        )
        restored_restore_count = _strict_int(
            restored_projection.get("restore_count"),
            field="bound restored Patch checkpoint restore_count",
            minimum=1,
        )
        if restored_restore_count != source_restore_count + 1:
            raise _fail("bound restored Patch checkpoint restore_count is not authoritative")
        restored_projection["restore_count"] = source_restore_count
        restored_authority = checkpoint_source_projection_authority_manifest(
            restored_projection
        )
        if restored_authority.get("source_projection_sha256") != source_projection_sha256:
            raise _fail("bound restored Patch checkpoint authority digest mismatch")
        projection = restored_projection

    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("restored native owner checkpoint metadata is malformed")
    projection_generation = _strict_int(
        metadata.get("generation"),
        field="restored native owner checkpoint generation",
        minimum=0,
    )
    projection_tick = _strict_int(
        metadata.get("tick"),
        field="restored native owner checkpoint tick",
        minimum=0,
    )
    if projection_generation != generation or projection_tick != tick:
        raise _fail("restored native owner checkpoint cursor contradicts its apply receipt")
    if metadata.get("agent_grid_runner_phase") != "generation_boundary":
        raise _fail("restored native owner checkpoint is not at a safe generation boundary")
    expected_epoch = f"generation:{generation}:tick:{tick}"
    if metadata.get("agent_grid_authority_epoch") != expected_epoch:
        raise _fail("restored native owner checkpoint authority epoch is inconsistent")

    pending_agent_ids = projection.get("pending_generation_free_agent_ids")
    pending_slots = projection.get("pending_generation_free_slots")
    if not isinstance(pending_agent_ids, (list, tuple)) or not isinstance(
        pending_slots, (list, tuple)
    ):
        raise _fail("restored native owner checkpoint pending-free fields are invalid")
    if pending_agent_ids or pending_slots:
        raise _fail("restored native owner checkpoint has pending generation frees")
    if expected_patch_profile is None:
        _validate_runner_projection_configuration(runner, metadata)
    else:
        _validate_patch_resume_projection_configuration(
            runner,
            metadata,
            expected_profile=expected_patch_profile,
        )
    apply_summary["source_projection_sha256"] = source_projection_sha256
    return projection, apply_summary


def _validate_row_cursor(
    value: Any,
    *,
    field: str,
    completed_generation: int,
    row_kind: str,
) -> None:
    cursor = _closed_dict(
        value,
        field=field,
        required_fields=frozenset({"completed_generation", "last_row_key"}),
    )
    if _strict_int(
        cursor["completed_generation"],
        field=f"{field}.completed_generation",
        minimum=0,
    ) != completed_generation:
        raise _fail(f"{field} generation contradicts the checkpoint")
    row_key = _strict_identifier(cursor["last_row_key"], field=f"{field}.last_row_key")
    prefix = f"generation:{completed_generation}:{row_kind}:"
    if not row_key.startswith(prefix) or not row_key[len(prefix) :]:
        raise _fail(f"{field}.last_row_key is outside the checkpoint frontier")


def _validate_cursor_range(value: Any, *, field: str) -> list[int]:
    if type(value) is not list or len(value) != 2:
        raise _fail(f"{field} must be a two-item array")
    start = _strict_int(value[0], field=f"{field}[0]", minimum=0)
    stop = _strict_int(value[1], field=f"{field}[1]", minimum=0)
    if stop < start:
        raise _fail(f"{field} must be a non-reversed range")
    return [start, stop]


def _validated_resume_identity(
    value: Any,
    *,
    generation: int,
    source_projection_sha256: str,
) -> dict[str, Any]:
    identity = _closed_dict(
        value,
        field="resume_envelope.architecture_b_identity",
        required_fields=_ARCHITECTURE_B_IDENTITY_FIELDS,
    )
    run_id = _strict_identifier(identity["run_id"], field="checkpoint run_id")
    controller_id = _strict_identifier(
        identity["controller_id"], field="checkpoint controller_id"
    )
    owner_epoch = _strict_int(
        identity["owner_epoch"], field="checkpoint owner_epoch", minimum=1
    )
    sequence = _strict_int(
        identity["snapshot_sequence"],
        field="checkpoint snapshot_sequence",
        minimum=1,
    )
    expected_snapshot_id = (
        f"{controller_id}:owner:{owner_epoch}:boundary:{sequence}"
    )
    if identity["snapshot_id"] != expected_snapshot_id:
        raise _fail("checkpoint snapshot identity is inconsistent")
    _strict_lowercase_sha256(
        identity["snapshot_digest_sha256"], field="checkpoint snapshot digest"
    )
    if identity["next_generation"] != generation or identity["tick"] != 0:
        raise _fail("checkpoint identity is not at generation:N:tick:0")
    if identity["phase"] != "generation_boundary":
        raise _fail("checkpoint identity is not at a complete-generation boundary")
    if identity["source_projection_sha256"] != source_projection_sha256:
        raise _fail("checkpoint identity source authority digest mismatch")
    copied = _dict_copy(identity, field="checkpoint architecture_b_identity")
    copied["run_id"] = run_id
    return copied


def _validated_observer_continuation(
    value: Any,
    *,
    generation: int,
    event_cursor: int,
    snapshot_sequence: int,
) -> dict[str, Any]:
    continuation = _closed_dict(
        value,
        field="resume_envelope.observer_continuation",
        required_fields=_OBSERVER_CONTINUATION_FIELDS,
    )
    if continuation["schema_version"] != NATIVE_OBSERVER_CONTINUATION_SCHEMA_VERSION:
        raise _fail("observer continuation schema is unsupported")
    scalar_expectations = {
        "checkpoint_next_generation": generation,
        "tick": 0,
        "typed_lifecycle_event_cursor": event_cursor,
        "exact_reporting_boundary_ordinal": snapshot_sequence,
        "chronicle_observation_ordinal": snapshot_sequence,
        "progress_sequence": snapshot_sequence,
    }
    for field, expected in scalar_expectations.items():
        actual = _strict_int(
            continuation[field], field=f"observer_continuation.{field}", minimum=0
        )
        if actual != expected:
            raise _fail(f"observer_continuation.{field} contradicts the checkpoint")

    if generation < 1:
        raise _fail("Architecture B resume requires at least one completed generation")
    completed_generation = generation - 1
    _validate_row_cursor(
        continuation["behavior_completed_generation_and_row_key"],
        field="observer_continuation.behavior_completed_generation_and_row_key",
        completed_generation=completed_generation,
        row_kind="agent",
    )
    if _strict_int(
        continuation["metrics_completed_generation"],
        field="observer_continuation.metrics_completed_generation",
        minimum=0,
    ) != completed_generation:
        raise _fail("observer continuation metrics cursor contradicts the checkpoint")
    _validate_row_cursor(
        continuation["lineage_completed_generation_and_row_key"],
        field="observer_continuation.lineage_completed_generation_and_row_key",
        completed_generation=completed_generation,
        row_kind="lineage",
    )

    schedule_phase = _closed_dict(
        continuation["schedule_phase"],
        field="observer_continuation.schedule_phase",
        required_fields=frozenset({"origin_generation", "next_generation"}),
    )
    schedule_origin = _strict_int(
        schedule_phase["origin_generation"],
        field="observer_continuation.schedule_phase.origin_generation",
        minimum=0,
    )
    if schedule_phase["next_generation"] != generation:
        raise _fail("observer continuation schedule next generation mismatch")
    if snapshot_sequence > generation:
        if schedule_origin != generation:
            raise _fail(
                "observer continuation absolute snapshot schedule phase is inconsistent"
            )
    elif schedule_origin + snapshot_sequence != generation:
        raise _fail("observer continuation schedule phase is inconsistent")

    deduplication = _closed_dict(
        continuation["deduplication"],
        field="observer_continuation.deduplication",
        required_fields=frozenset({"acknowledged_record_keys"}),
    )
    record_keys = deduplication["acknowledged_record_keys"]
    if type(record_keys) is not list or len(record_keys) > 4096:
        raise _fail("observer continuation deduplication keys are invalid")
    if len(set(record_keys)) != len(record_keys):
        raise _fail("observer continuation deduplication keys are not unique")
    for index, record_key in enumerate(record_keys):
        _strict_identifier(
            record_key,
            field=f"observer_continuation.deduplication.acknowledged_record_keys[{index}]",
        )

    ranges = _closed_dict(
        continuation["acknowledged_channel_cursor_ranges"],
        field="observer_continuation.acknowledged_channel_cursor_ranges",
        required_fields=_ACKNOWLEDGED_CHANNELS,
    )
    normalized_ranges = {
        channel: _validate_cursor_range(
            ranges[channel],
            field=f"observer_continuation.acknowledged_channel_cursor_ranges.{channel}",
        )
        for channel in sorted(_ACKNOWLEDGED_CHANNELS)
    }
    generation_range = [completed_generation, generation]
    ordinal_range = [snapshot_sequence - 1, snapshot_sequence]
    expected_ranges = {
        "exact_reporting": generation_range,
        "behaviors": generation_range,
        "metrics": generation_range,
        "lineage": generation_range,
        "chronicle": ordinal_range,
        "checkpoint_generation": [generation, generation + 1],
    }
    for channel, expected in expected_ranges.items():
        if normalized_ranges[channel] != expected:
            raise _fail(f"observer continuation {channel} cursor range mismatch")
    if normalized_ranges["typed_lifecycle"][1] != event_cursor:
        raise _fail("observer continuation typed lifecycle cursor range mismatch")
    return _dict_copy(continuation, field="observer_continuation")


def _phase4_capabilities(
    module: Any,
    *,
    required: bool = True,
) -> frozenset[str]:
    raw = getattr(module, NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE, None)
    if isinstance(raw, Mapping):
        capabilities = frozenset(
            str(name) for name, enabled in raw.items() if enabled is True
        )
    elif isinstance(raw, (list, tuple, set, frozenset)):
        capabilities = frozenset(str(name) for name in raw)
    else:
        capabilities = frozenset()
    missing = sorted(NATIVE_PHASE4_REQUIRED_CAPABILITIES - capabilities)
    if required and missing:
        raise _fail(
            "prebuilt extension Architecture B Phase 4 protocol capabilities "
            f"are missing: {missing}"
        )
    return capabilities


def _construct_native_controller(
    controller_type: Any,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    run_id: str,
    *,
    persistence_mailbox_required: bool,
    checkpoint_projection: Mapping[str, Any] | None = None,
) -> Any:
    arguments = (tensors, metadata, run_id, persistence_mailbox_required)
    if checkpoint_projection is not None:
        arguments = (*arguments, checkpoint_projection)
    try:
        signature = inspect.signature(controller_type)
    except (TypeError, ValueError):
        if checkpoint_projection is not None:
            return controller_type(*arguments)
        if persistence_mailbox_required:
            return controller_type(*arguments)
        # Pybind classes do not expose an inspectable signature.  The accepted
        # Phase 3 controller has the original three-argument constructor, while
        # Phase 4 keeps the fourth mailbox argument optional when disabled.
        return controller_type(tensors, metadata, run_id)
    try:
        signature.bind(*arguments)
    except TypeError as exc:
        if checkpoint_projection is not None:
            raise _fail(
                "native controller constructor lacks checkpoint runtime prefix import"
            ) from exc
        if persistence_mailbox_required:
            raise _fail(
                "native controller constructor lacks persistence mailbox mode"
            ) from exc
        return controller_type(tensors, metadata, run_id)
    return controller_type(*arguments)


def _validate_phase4_owner_protocol(native_controller: Any) -> None:
    required_methods = (
        "run_generations",
        "read_snapshots",
        "acknowledge_snapshot_persistence",
        "fail_snapshot_persistence",
        "request_cancel",
        "prepare_snapshot_validation",
        "commit_snapshot_validation",
        "acknowledge_snapshot_validation",
        "fail_snapshot_validation",
        "final_result",
        "controller_manifest",
    )
    missing = [
        name
        for name in required_methods
        if not callable(getattr(native_controller, name, None))
    ]
    if missing:
        raise _fail(f"native Phase 4 owner protocol methods are missing: {missing}")


def _validate_snapshot_schedule(value: Any) -> dict[str, Any]:
    schedule = _dict_copy(value, field="snapshot_schedule")
    actual_fields = frozenset(schedule)
    missing = sorted(_SNAPSHOT_SCHEDULE_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _SNAPSHOT_SCHEDULE_FIELDS)
    if missing or unexpected:
        raise _fail(
            "snapshot_schedule fields mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    checkpoint_on_cancel = schedule["checkpoint_on_cancel"]
    if type(checkpoint_on_cancel) is not bool:
        raise _fail("snapshot_schedule.checkpoint_on_cancel must be boolean")
    if checkpoint_on_cancel is not True:
        raise _fail("Architecture B Phase 4 requires checkpoint_on_cancel=true")
    if schedule["schema_version"] != NATIVE_SNAPSHOT_SCHEDULE_SCHEMA_VERSION:
        raise _fail("snapshot schedule schema is unsupported")
    progress = _strict_int(
        schedule["progress_every_generations"],
        field="snapshot_schedule.progress_every_generations",
        minimum=0,
    )
    report = _strict_int(
        schedule["report_every_generations"],
        field="snapshot_schedule.report_every_generations",
        minimum=0,
    )
    checkpoint = _strict_int(
        schedule["checkpoint_every_generations"],
        field="snapshot_schedule.checkpoint_every_generations",
        minimum=0,
    )
    if progress not in {0, 1}:
        raise _fail("snapshot progress cadence supports only disabled or every generation")
    if report not in {0, 1, 2, 8}:
        raise _fail("snapshot report cadence is outside the closed Phase 4 envelope")
    if checkpoint not in {0, 1, 2, 8}:
        raise _fail("snapshot checkpoint cadence is outside the closed Phase 4 envelope")
    raw_explicit = schedule["explicit_next_generations"]
    if not isinstance(raw_explicit, list):
        raise _fail("snapshot explicit checkpoint generations must be a list")
    explicit = [
        _strict_int(
            value,
            field=f"snapshot_schedule.explicit_next_generations[{index}]",
            minimum=0,
        )
        for index, value in enumerate(raw_explicit)
    ]
    if explicit != sorted(set(explicit)):
        raise _fail(
            "snapshot explicit checkpoint generations must be sorted and unique"
        )
    if progress == 1 and report != 0:
        raise _fail("snapshot lightweight progress and periodic exact reporting conflict")
    if schedule["emit_final_snapshot"] is not True:
        raise _fail("snapshot schedule must emit a final snapshot")
    if schedule["typed_lifecycle_mode"] != "cursor_bounded_delta":
        raise _fail("snapshot typed lifecycle mode is unsupported")
    if schedule["mailbox_capacity"] != 1:
        raise _fail("snapshot mailbox capacity must equal one")
    normalized = _dict_copy(schedule, field="snapshot_schedule")
    normalized["explicit_next_generations"] = explicit
    return normalized


def _validate_architecture_b_envelope(runner: Any) -> None:
    env_cfg = getattr(runner, "env_cfg", None)
    tier4_enabled = bool(
        getattr(env_cfg, "tier4_juvenile_vulnerability_enable", False)
    )
    try:
        tier4_juvenile_steps = int(
            getattr(env_cfg, "tier4_juvenile_steps", 0) or 0
        )
    except (TypeError, ValueError, OverflowError):
        tier4_juvenile_steps = 1
    if tier4_enabled or tier4_juvenile_steps > 0:
        raise _fail(
            f"{TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_REASON}: "
            f"{TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_MESSAGE}"
        )


def _tensor_manifest_bytes(projection: Mapping[str, Any], *, field: str) -> int:
    tensors = projection.get("tensors")
    manifest = projection.get("tensor_manifest")
    if not isinstance(tensors, Mapping) or not isinstance(manifest, Mapping):
        raise _fail(f"{field} is missing tensors or tensor_manifest")
    if set(tensors) != set(manifest):
        raise _fail(f"{field} tensor_manifest fields do not match tensors")

    total = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        row = manifest[name]
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise _fail(f"{field} has a malformed tensor_manifest row")
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            raise _fail(f"{field}.tensors.{name} is not tensor-like")
        exact_bytes = int(tensor.numel()) * int(tensor.element_size())
        recorded_bytes = _strict_int(
            row.get("bytes"),
            field=f"{field}.tensor_manifest.{name}.bytes",
            minimum=0,
        )
        if recorded_bytes != exact_bytes:
            raise _fail(
                f"{field}.tensor_manifest.{name}.bytes mismatch: "
                f"recorded={recorded_bytes} exact={exact_bytes}"
            )
        total += exact_bytes
    return total


def _projection_digest_manifests(projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_projection": checkpoint_source_projection_digest_manifest(projection),
        "trajectory_state": checkpoint_digest_manifest(
            projection,
            layer=DigestLayer.TRAJECTORY_STATE,
        ),
        "canonical_checkpoint_storage": checkpoint_digest_manifest(
            projection,
            layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
        ),
        "diagnostic_projection": checkpoint_digest_manifest(
            projection,
            layer=DigestLayer.DIAGNOSTIC_PROJECTION,
        ),
    }


def _patch_initial_import_metrics(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    pending_generation_free_agent_ids: Any,
    pending_generation_free_slots: Any,
) -> dict[str, int]:
    from .architecture_b_crossing_metrics import (
        architecture_b_crossing_payload_metrics,
    )

    try:
        metrics = architecture_b_crossing_payload_metrics(
            {
                "tensors": tensors,
                "metadata": metadata,
                "pending_generation_free_agent_ids": (
                    pending_generation_free_agent_ids
                ),
                "pending_generation_free_slots": pending_generation_free_slots,
            }
        )
    except Exception as exc:
        raise _fail(
            f"Patch instrumentation could not account initial import bytes: {exc}"
        ) from exc
    return {
        "represented_payload_bytes": _strict_int(
            metrics.get("represented_payload_bytes"),
            field="Patch initial import represented_payload_bytes",
            minimum=0,
        ),
        "unique_tensor_storage_bytes": _strict_int(
            metrics.get("unique_tensor_storage_bytes"),
            field="Patch initial import unique_tensor_storage_bytes",
            minimum=0,
        ),
        "unique_tensor_storage_count": _strict_int(
            metrics.get("unique_tensor_storage_count"),
            field="Patch initial import unique_tensor_storage_count",
            minimum=0,
        ),
    }


class NativeRunController:
    """Thin Python facade for one CPU Architecture B native dispatch."""

    def __init__(
        self,
        native_controller: Any,
        *,
        start_generation: int,
        default_steps_per_generation: int,
        default_max_births: int,
        safe_default_decision: Mapping[str, Any],
        extension_path: str,
        run_id: str,
        persistence_mailbox_required: bool = False,
        initial_boundary_ordinal: int = 0,
        schedule_origin_generation: int | None = None,
        phase4_capabilities: frozenset[str] = frozenset(),
        architecture_world: str = "flat",
        patch_profile: str | None = None,
        patch_extension_capabilities: Mapping[str, bool] | None = None,
        patch_instrumentation_enabled: bool = False,
        patch_initial_import_metrics: Mapping[str, Any] | None = None,
        compact_reporting_only: bool = False,
        phase_e_passive_no_churn_guard: Mapping[str, Any] | None = None,
    ) -> None:
        self._native_controller = native_controller
        self._start_generation = int(start_generation)
        self._default_steps_per_generation = int(default_steps_per_generation)
        self._default_max_births = int(default_max_births)
        self._phase_e_passive_no_churn_guard = (
            _dict_copy(
                phase_e_passive_no_churn_guard,
                field="Phase E passive modular no-churn guard",
            )
            if phase_e_passive_no_churn_guard is not None
            else None
        )
        if (
            self._phase_e_passive_no_churn_guard is not None
            and self._default_max_births != 0
        ):
            raise _fail("Phase E passive modular milestone requires max_births=0")
        self._safe_default_decision = _dict_copy(
            safe_default_decision,
            field="safe_default_decision",
        )
        self._extension_path = str(extension_path)
        self._persistence_mailbox_required = bool(persistence_mailbox_required)
        self._initial_boundary_ordinal = _strict_int(
            initial_boundary_ordinal,
            field="initial_boundary_ordinal",
            minimum=0,
        )
        self._schedule_origin_generation = _strict_int(
            self._start_generation
            if schedule_origin_generation is None
            else schedule_origin_generation,
            field="schedule_origin_generation",
            minimum=0,
        )
        if self._schedule_origin_generation > self._start_generation:
            raise _fail("schedule origin exceeds the imported generation")
        self._phase4_capabilities = frozenset(phase4_capabilities)
        if type(patch_instrumentation_enabled) is not bool:
            raise _fail("patch_instrumentation_enabled must be boolean")
        if type(compact_reporting_only) is not bool:
            raise _fail("compact_reporting_only must be boolean")
        self._compact_reporting_only = compact_reporting_only
        if architecture_world not in {"flat", "patch"}:
            raise _fail("Architecture B controller world must be flat or patch")
        if architecture_world == "patch":
            if patch_profile not in SUPPORTED_PATCH_PROFILES:
                raise _fail("Patch Architecture B controller profile is unsupported")
            if not isinstance(patch_extension_capabilities, Mapping):
                raise _fail("Patch Architecture B controller lacks capability receipts")
            if patch_instrumentation_enabled and not isinstance(
                patch_initial_import_metrics, Mapping
            ):
                raise _fail(
                    "enabled Patch instrumentation lacks initial import metrics"
                )
            if not patch_instrumentation_enabled and patch_initial_import_metrics is not None:
                raise _fail(
                    "disabled Patch instrumentation received initial import metrics"
                )
        elif (
            patch_profile is not None
            or patch_extension_capabilities is not None
            or patch_instrumentation_enabled
            or patch_initial_import_metrics is not None
        ):
            raise _fail("Flat Architecture B controller received Patch configuration")
        self._architecture_world = architecture_world
        self._patch_profile = patch_profile
        self._patch_extension_capabilities = (
            _dict_copy(
                patch_extension_capabilities,
                field="Patch extension capabilities",
            )
            if patch_extension_capabilities is not None
            else None
        )
        self._patch_instrumentation_enabled = patch_instrumentation_enabled
        self._patch_initial_import_metrics: dict[str, int] | None = None
        if patch_initial_import_metrics is not None:
            initial_metrics = _closed_dict(
                _dict_copy(
                    patch_initial_import_metrics,
                    field="Patch initial import metrics",
                ),
                field="Patch initial import metrics",
                required_fields=frozenset(
                    {
                        "represented_payload_bytes",
                        "unique_tensor_storage_bytes",
                        "unique_tensor_storage_count",
                    }
                ),
            )
            self._patch_initial_import_metrics = {
                field: _strict_int(
                    initial_metrics[field],
                    field=f"Patch initial import metrics.{field}",
                    minimum=0,
                )
                for field in initial_metrics
            }
        self._patch_native_instrumentation: dict[str, int] | None = None
        expected_run_id = _strict_identifier(run_id, field="run_id")
        self._trajectory_dispatch_count = 0
        self._boundary_tensor_manifest_bytes = 0
        self._final: dict[str, Any] | None = None
        self._failure: str = ""
        self._accepted_cancel_identity: tuple[str, int] | None = None
        self._accepted_cancel_reason: str | None = None
        initial_manifest = self._native_manifest()
        self._run_id = _strict_identifier(
            initial_manifest.get("run_id"),
            field="native run_id",
        )
        if self._run_id != expected_run_id:
            raise _fail("native controller run identity mismatch")
        self._controller_id = _strict_identifier(
            initial_manifest.get("controller_id"),
            field="native controller_id",
        )
        self._owner_epoch = _strict_int(
            initial_manifest.get("owner_epoch"),
            field="native owner_epoch",
            minimum=1,
        )
        self._validate_native_manifest(
            expected_status="initialized",
            manifest=initial_manifest,
        )

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        generation: int = 0,
        tick: int = 0,
        extension_path: str | Path | None = None,
        expected_extension_sha256: str | None = None,
        persistence_mailbox_required: bool = False,
        patch_instrumentation_enabled: bool = False,
        compact_reporting_only: bool = False,
    ) -> "NativeRunController":
        if type(persistence_mailbox_required) is not bool:
            raise _fail("persistence_mailbox_required must be boolean")
        if type(patch_instrumentation_enabled) is not bool:
            raise _fail("patch_instrumentation_enabled must be boolean")
        if type(compact_reporting_only) is not bool:
            raise _fail("compact_reporting_only must be boolean")
        # Compact mode retains small native summaries and requires an explicit
        # terminal report or checkpoint for the final full projection.
        start_generation = _strict_int(generation, field="generation", minimum=0)
        start_tick = _strict_int(tick, field="tick", minimum=0)
        if start_tick != 0:
            raise _fail("Architecture B import requires tick zero")

        architecture_world, patch_profile, patch_classification = (
            _classify_architecture_b_world_profile(runner)
        )
        if patch_instrumentation_enabled and architecture_world != "patch":
            raise _fail("Patch instrumentation cannot be enabled for a Flat controller")
        patch_extension_path: Path | None = None
        patch_extension_sha256: str | None = None
        decision: Any | None
        if architecture_world == "flat":
            decision = validate_native_runtime_supported(runner)
            if not decision.extension_loaded:
                raise _fail("qualified prebuilt extension is not loaded")
            if decision.extension_runtime_jit or not decision.runtime_jit_prohibited:
                raise _fail("runtime JIT is not prohibited by the qualification")
        else:
            decision = None
            patch_extension_path, patch_extension_sha256 = (
                _explicit_patch_extension_artifact(
                    extension_path,
                    expected_extension_sha256,
                )
            )

        _validate_architecture_b_envelope(runner)

        restored_import = _restored_native_checkpoint_projection(
            runner,
            generation=start_generation,
            tick=start_tick,
            expected_patch_profile=(
                patch_profile if architecture_world == "patch" else None
            ),
        )
        checkpoint_projection_for_import: Mapping[str, Any] | None = None
        if restored_import is None:
            from perf.cxxbackbone10_resident_state_owner import build_import_projection

            runner_projection = build_import_projection(
                runner,
                {
                    "flat": architecture_world == "flat",
                    "patch": architecture_world == "patch",
                    "churn": True,
                    "cross_path_overlay": bool(
                        getattr(
                            getattr(runner, "env", None),
                            "cfg",
                            None,
                        )
                        and getattr(
                            runner.env.cfg,
                            "reproduce_requires_mate",
                            False,
                        )
                    ),
                    **(
                        {"wdl": True}
                        if architecture_world == "patch"
                        and patch_classification is not None
                        and patch_classification.get("wdl_native_execution", False)
                        else {}
                    ),
                    **(
                        {"jungle": True}
                        if architecture_world == "patch"
                        and patch_classification is not None
                        and patch_classification.get("jungle_native_execution", False)
                        else {}
                    ),
                },
                generation=start_generation,
                tick=start_tick,
            )
            if (
                runner_projection.pending_generation_free_agent_ids
                or runner_projection.pending_generation_free_slots
            ):
                raise _fail("Architecture B initial import has pending generation frees")
            projection_tensors = runner_projection.tensors
            projection_metadata = runner_projection.metadata
            pending_generation_free_agent_ids = (
                runner_projection.pending_generation_free_agent_ids
            )
            pending_generation_free_slots = (
                runner_projection.pending_generation_free_slots
            )
            restored_apply_summary = None
        else:
            checkpoint_projection, restored_apply_summary = restored_import
            projection_tensors = checkpoint_projection["tensors"]
            projection_metadata = checkpoint_projection["metadata"]
            pending_generation_free_agent_ids = checkpoint_projection[
                "pending_generation_free_agent_ids"
            ]
            pending_generation_free_slots = checkpoint_projection[
                "pending_generation_free_slots"
            ]
            checkpoint_projection_for_import = checkpoint_projection

        patch_initial_import_metrics: dict[str, int] | None = None
        if architecture_world == "patch":
            if patch_profile is None:
                raise _fail("Patch Architecture B classification lost its profile")
            _validate_patch_import_projection(
                projection_metadata,
                expected_profile=patch_profile,
            )
            if patch_instrumentation_enabled:
                patch_initial_import_metrics = _patch_initial_import_metrics(
                    projection_tensors,
                    projection_metadata,
                    pending_generation_free_agent_ids=(
                        pending_generation_free_agent_ids
                    ),
                    pending_generation_free_slots=pending_generation_free_slots,
                )

        if architecture_world == "patch":
            assert patch_extension_path is not None
            assert patch_extension_sha256 is not None
            selected_extension_path = patch_extension_path
            selected_extension_sha256 = patch_extension_sha256
        else:
            selected_extension_path = Path(
                decision.extension_path if extension_path is None else extension_path
            ).expanduser().resolve()
            if not selected_extension_path.is_file():
                raise _fail(
                    f"Architecture B extension does not exist: {selected_extension_path}"
                )
            selected_extension_sha256 = _sha256(selected_extension_path)
            if extension_path is not None:
                expected_sha = str(expected_extension_sha256 or "").strip().lower()
                if len(expected_sha) != 64 or any(
                    char not in "0123456789abcdef" for char in expected_sha
                ):
                    raise _fail(
                        "explicit Architecture B extension requires a lowercase SHA-256"
                    )
                if selected_extension_sha256 != expected_sha:
                    raise _fail(
                        "explicit Architecture B extension SHA-256 mismatch: "
                        f"expected={expected_sha} actual={selected_extension_sha256}"
                    )

        module = load_prebuilt_extension(selected_extension_path)
        controller_type = getattr(module, NATIVE_CONTROLLER_CLASS, None)
        if controller_type is None or not callable(controller_type):
            raise _fail(f"prebuilt extension is missing {NATIVE_CONTROLLER_CLASS}")
        controller_version = str(
            getattr(module, "EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION", "") or ""
        )
        if controller_version != NATIVE_CONTROLLER_VERSION:
            raise _fail(
                "prebuilt extension Architecture B controller version mismatch: "
                f"{controller_version or '<missing>'}"
            )
        phase4_capabilities = _phase4_capabilities(
            module,
            required=persistence_mailbox_required,
        )
        patch_capabilities: dict[str, bool] | None = None
        if architecture_world == "patch":
            if NATIVE_INTER_TICK_CONTROL_CAPABILITY not in phase4_capabilities:
                raise _fail(
                    "prebuilt extension lacks the declared native inter-tick control window"
                )
            patch_capabilities = _patch_extension_capabilities(module)

        metadata = copy.deepcopy(projection_metadata)
        metadata["native_runtime_architecture_b_compact_reporting_only_v1"] = (
            compact_reporting_only
        )
        run_id = _logical_run_id(runner)
        native_controller = _construct_native_controller(
            controller_type,
            projection_tensors,
            metadata,
            run_id,
            persistence_mailbox_required=persistence_mailbox_required,
            checkpoint_projection=checkpoint_projection_for_import,
        )
        if persistence_mailbox_required:
            _validate_phase4_owner_protocol(native_controller)
        if architecture_world == "patch" and restored_apply_summary is not None:
            resumed_manifest = _dict_copy(
                native_controller.controller_manifest(),
                field="resumed Patch native controller manifest",
            )
            if resumed_manifest.get("checkpoint_runtime_prefix_imported") is not True:
                raise _fail("resumed Patch native owner did not import one checkpoint prefix")
        sim_cfg = getattr(runner, "sim_cfg", None)
        steps_per_generation = _strict_int(
            getattr(sim_cfg, "steps_per_generation", 0),
            field="steps_per_generation",
            minimum=1,
        )
        max_births = _strict_int(
            getattr(sim_cfg, "native_runtime_max_births_per_generation", 1),
            field="native_runtime_max_births_per_generation",
            minimum=0,
        )
        phase_e_passive_guard = _phase_e_passive_no_churn_guard_for_controller(
            sim_cfg,
            {
                "metadata": projection_metadata,
                "tensors": projection_tensors,
            },
            max_births=max_births,
        )
        if decision is not None:
            decision_payload = decision.to_dict()
        else:
            if patch_classification is None:
                raise _fail("Patch Architecture B classification receipt is missing")
            decision_payload = {
                "schema_version": (
                    NATIVE_PATCH_CONTROLLER_QUALIFICATION_SCHEMA_VERSION
                ),
                "controller_integration_supported": True,
                "controller_execution_active": True,
                "production_routing_enabled": False,
                "safe_default_promoted": False,
                "promotion_accepted": False,
                "cuda_supported": False,
                "runtime_jit_prohibited": True,
                "hidden_fallback_allowed": False,
                "instrumentation_enabled": patch_instrumentation_enabled,
                "world": "patch",
                "profile": patch_profile,
                "profile_classification": _detached_copy(patch_classification),
            }
        decision_payload["architecture_b_extension_path"] = str(selected_extension_path)
        decision_payload["architecture_b_extension_sha256"] = selected_extension_sha256
        decision_payload["architecture_b_extension_explicit_development_artifact"] = extension_path is not None
        if restored_apply_summary is not None:
            decision_payload["architecture_b_checkpoint_resume"] = True
            decision_payload["architecture_b_checkpoint_import_source"] = (
                NATIVE_CHECKPOINT_APPLY_SOURCE
            )
            decision_payload["architecture_b_source_projection_sha256"] = (
                restored_apply_summary["source_projection_sha256"]
            )
        return cls(
            native_controller,
            start_generation=start_generation,
            default_steps_per_generation=steps_per_generation,
            default_max_births=max_births,
            safe_default_decision=decision_payload,
            extension_path=str(selected_extension_path),
            run_id=run_id,
            persistence_mailbox_required=persistence_mailbox_required,
            initial_boundary_ordinal=0,
            schedule_origin_generation=start_generation,
            phase4_capabilities=phase4_capabilities,
            architecture_world=architecture_world,
            patch_profile=patch_profile,
            patch_extension_capabilities=patch_capabilities,
            patch_instrumentation_enabled=patch_instrumentation_enabled,
            patch_initial_import_metrics=patch_initial_import_metrics,
            compact_reporting_only=compact_reporting_only,
            phase_e_passive_no_churn_guard=phase_e_passive_guard,
        )

    @classmethod
    def from_checkpoint_projection(
        cls,
        runner: Any,
        resume_envelope: Mapping[str, Any],
        *,
        extension_path: str | Path | None = None,
        expected_extension_sha256: str | None = None,
    ) -> "NativeRunController":
        architecture_world, patch_profile, patch_classification = (
            _classify_architecture_b_world_profile(runner)
        )
        patch_resume = architecture_world == "patch"

        patch_extension_path: Path | None = None
        patch_extension_sha256: str | None = None
        if patch_resume:
            patch_extension_path, patch_extension_sha256 = (
                _explicit_patch_extension_artifact(
                    extension_path,
                    expected_extension_sha256,
                )
            )

        try:
            rebound_resume_envelope = (
                validate_architecture_b_resume_envelope_authority(resume_envelope)
            )
        except Exception as exc:
            raise _fail(
                f"checkpoint resume envelope lacks strict bundle authority: {exc}"
            ) from exc
        envelope = _closed_dict(
            rebound_resume_envelope,
            field="resume_envelope",
            required_fields=_RESUME_ENVELOPE_FIELDS,
        )
        if envelope["schema_version"] != (
            NATIVE_CHECKPOINT_RESUME_ENVELOPE_SCHEMA_VERSION
        ):
            raise _fail("Architecture B checkpoint resume envelope schema mismatch")
        if envelope["bundle_kind"] != "architecture_b":
            raise _fail("Architecture B checkpoint resume envelope kind mismatch")

        decision: Any | None
        if patch_resume:
            decision = None
        else:
            decision = validate_native_runtime_supported(runner)
            if not decision.extension_loaded:
                raise _fail("qualified prebuilt extension is not loaded")
            if decision.extension_runtime_jit or not decision.runtime_jit_prohibited:
                raise _fail("runtime JIT is not prohibited by the qualification")
        _validate_architecture_b_envelope(runner)

        projection = _dict_copy(
            envelope["checkpoint_projection"],
            field="resume_envelope.checkpoint_projection",
        )
        try:
            validate_native_checkpoint_projection(projection)
            source_authority = checkpoint_source_projection_authority_manifest(
                projection
            )
        except Exception as exc:
            raise _fail(f"checkpoint projection authority validation failed: {exc}") from exc
        source_projection_sha256 = _strict_lowercase_sha256(
            envelope["source_projection_sha256"],
            field="resume_envelope.source_projection_sha256",
        )
        if source_authority.get("source_projection_sha256") != (
            source_projection_sha256
        ):
            raise _fail("checkpoint source projection authority digest mismatch")

        metadata = _dict_copy(
            projection.get("metadata"),
            field="resume checkpoint metadata",
        )
        generation = _strict_int(
            metadata.get("generation"),
            field="resume checkpoint generation",
            minimum=1,
        )
        tick = _strict_int(
            metadata.get("tick"), field="resume checkpoint tick", minimum=0
        )
        if tick != 0:
            raise _fail("Architecture B resume requires generation:N:tick:0")
        if metadata.get("agent_grid_runner_phase") != "generation_boundary":
            raise _fail("Architecture B resume projection is not at a safe boundary")
        pending_agent_ids = projection.get("pending_generation_free_agent_ids")
        pending_slots = projection.get("pending_generation_free_slots")
        if not isinstance(pending_agent_ids, (list, tuple)) or not isinstance(
            pending_slots, (list, tuple)
        ):
            raise _fail("Architecture B resume projection pending-free fields are invalid")
        if pending_agent_ids or pending_slots:
            raise _fail("Architecture B resume projection has pending generation frees")

        identity = _validated_resume_identity(
            envelope["architecture_b_identity"],
            generation=generation,
            source_projection_sha256=source_projection_sha256,
        )
        try:
            from .lifecycle_events import (
                typed_lifecycle_events_from_projection,
                validated_typed_lifecycle_event_cursor,
            )

            event_cursor = validated_typed_lifecycle_event_cursor(
                projection,
                typed_lifecycle_events_from_projection(projection),
            )
        except Exception as exc:
            raise _fail(
                f"resume checkpoint typed lifecycle cursor is invalid: {exc}"
            ) from exc
        continuation = _validated_observer_continuation(
            envelope["observer_continuation"],
            generation=generation,
            event_cursor=event_cursor,
            snapshot_sequence=int(identity["snapshot_sequence"]),
        )

        expected_import_metadata = _dict_copy(
            metadata, field="expected controller import metadata"
        )
        expected_import_metadata["architecture_b_identity"] = _detached_copy(identity)
        expected_import_metadata["observer_continuation"] = _detached_copy(
            continuation
        )
        supplied_import_metadata = _dict_copy(
            envelope["controller_import_metadata"],
            field="resume_envelope.controller_import_metadata",
        )
        if not _exact_values_equal(supplied_import_metadata, expected_import_metadata):
            raise _fail("checkpoint controller import metadata contradicts its envelope")

        if patch_resume:
            if patch_profile is None:
                raise _fail("Patch Architecture B classification lost its profile")
            steps_per_generation, max_births = (
                _validate_patch_resume_projection_configuration(
                    runner,
                    metadata,
                    expected_profile=patch_profile,
                )
            )
        else:
            steps_per_generation, max_births = _validate_runner_projection_configuration(
                runner, metadata
            )
        phase_e_passive_guard = _phase_e_passive_no_churn_guard_for_controller(
            getattr(runner, "sim_cfg", None),
            projection,
            max_births=max_births,
        )
        configured_run_id = str(
            getattr(getattr(runner, "sim_cfg", None), "chronicle_run_id", "") or ""
        ).strip()
        run_id = str(identity["run_id"])
        if configured_run_id and configured_run_id != run_id:
            raise _fail("checkpoint run identity contradicts the qualified runner")

        if patch_resume:
            assert patch_extension_path is not None
            assert patch_extension_sha256 is not None
            selected_extension_path = patch_extension_path
            selected_extension_sha256 = patch_extension_sha256
        else:
            selected_extension_path = Path(
                decision.extension_path if extension_path is None else extension_path
            ).expanduser().resolve()
            if not selected_extension_path.is_file():
                raise _fail(
                    f"Architecture B extension does not exist: {selected_extension_path}"
                )
            selected_extension_sha256 = _sha256(selected_extension_path)
            if extension_path is not None:
                expected_sha = _strict_lowercase_sha256(
                    expected_extension_sha256,
                    field="explicit Architecture B extension SHA-256",
                )
                if selected_extension_sha256 != expected_sha:
                    raise _fail(
                        "explicit Architecture B extension SHA-256 mismatch: "
                        f"expected={expected_sha} actual={selected_extension_sha256}"
                    )

        module = load_prebuilt_extension(selected_extension_path)
        controller_type = getattr(module, NATIVE_CONTROLLER_CLASS, None)
        if controller_type is None or not callable(controller_type):
            raise _fail(f"prebuilt extension is missing {NATIVE_CONTROLLER_CLASS}")
        controller_version = str(
            getattr(module, "EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION", "")
            or ""
        )
        if controller_version != NATIVE_CONTROLLER_VERSION:
            raise _fail(
                "prebuilt extension Architecture B controller version mismatch: "
                f"{controller_version or '<missing>'}"
            )
        phase4_capabilities = _phase4_capabilities(module)
        patch_capabilities: dict[str, bool] | None = None
        projection_for_controller = projection
        if patch_resume:
            if NATIVE_INTER_TICK_CONTROL_CAPABILITY not in phase4_capabilities:
                raise _fail(
                    "prebuilt extension lacks the declared native inter-tick control window"
                )
            patch_capabilities = _patch_extension_capabilities(module)
            assert patch_profile is not None
            projection_for_controller = _restore_patch_checkpoint_projection(
                projection,
                extension_path=selected_extension_path,
                expected_profile=patch_profile,
                source_projection_sha256=source_projection_sha256,
            )

        native_metadata = _dict_copy(
            supplied_import_metadata,
            field="native resume import metadata",
        )
        native_controller = _construct_native_controller(
            controller_type,
            projection_for_controller["tensors"],
            native_metadata,
            run_id,
            persistence_mailbox_required=True,
            checkpoint_projection=projection_for_controller,
        )
        _validate_phase4_owner_protocol(native_controller)

        if patch_resume:
            resumed_manifest = _dict_copy(
                native_controller.controller_manifest(),
                field="resumed Patch native controller manifest",
            )
            previous_owner_epoch = int(identity["owner_epoch"])
            if resumed_manifest.get("owner_epoch") != previous_owner_epoch + 1:
                raise _fail("resumed Patch native owner epoch is not fresh")
            if resumed_manifest.get("controller_id") == identity["controller_id"]:
                raise _fail("resumed Patch native controller identity is not fresh")
            if resumed_manifest.get("checkpoint_runtime_prefix_imported") is not True:
                raise _fail("resumed Patch native owner did not import one checkpoint prefix")

        if patch_resume:
            if patch_classification is None:
                raise _fail("Patch Architecture B classification receipt is missing")
            decision_payload = {
                "schema_version": (
                    NATIVE_PATCH_CONTROLLER_QUALIFICATION_SCHEMA_VERSION
                ),
                "controller_integration_supported": True,
                "controller_execution_active": True,
                "production_routing_enabled": False,
                "safe_default_promoted": False,
                "promotion_accepted": False,
                "cuda_supported": False,
                "runtime_jit_prohibited": True,
                "hidden_fallback_allowed": False,
                "instrumentation_enabled": False,
                "world": "patch",
                "profile": patch_profile,
                "profile_classification": _detached_copy(patch_classification),
                "checkpoint_restore_engine": "NativeEielEngine.from_checkpoint_projection",
            }
        else:
            decision_payload = decision.to_dict()
        decision_payload["architecture_b_extension_path"] = str(
            selected_extension_path
        )
        decision_payload["architecture_b_extension_sha256"] = (
            selected_extension_sha256
        )
        decision_payload[
            "architecture_b_extension_explicit_development_artifact"
        ] = extension_path is not None
        decision_payload["architecture_b_checkpoint_resume"] = True
        decision_payload["architecture_b_source_projection_sha256"] = (
            source_projection_sha256
        )
        schedule_phase = continuation["schedule_phase"]
        assert isinstance(schedule_phase, Mapping)
        return cls(
            native_controller,
            start_generation=generation,
            default_steps_per_generation=steps_per_generation,
            default_max_births=max_births,
            safe_default_decision=decision_payload,
            extension_path=str(selected_extension_path),
            run_id=run_id,
            persistence_mailbox_required=True,
            initial_boundary_ordinal=int(identity["snapshot_sequence"]),
            schedule_origin_generation=int(schedule_phase["origin_generation"]),
            phase4_capabilities=phase4_capabilities,
            architecture_world=architecture_world,
            patch_profile=patch_profile,
            patch_extension_capabilities=patch_capabilities,
            phase_e_passive_no_churn_guard=phase_e_passive_guard,
        )

    def run_many(
        self,
        generation_count: int,
        *,
        steps_per_generation: int | None = None,
        max_births: int | None = None,
    ) -> dict[str, Any]:
        if self._trajectory_dispatch_count:
            raise _fail("Architecture B controller is single-use")
        count = _strict_int(generation_count, field="generation_count", minimum=1)
        steps = _strict_int(
            self._default_steps_per_generation
            if steps_per_generation is None
            else steps_per_generation,
            field="steps_per_generation",
            minimum=1,
        )
        births = _strict_int(
            self._default_max_births if max_births is None else max_births,
            field="max_births",
            minimum=0,
        )
        if self._phase_e_passive_no_churn_guard is not None:
            if count != 1:
                raise _fail("Phase E passive modular milestone requires one generation")
            if births != 0:
                raise _fail("Phase E passive modular milestone requires max_births=0")

        self._trajectory_dispatch_count = 1
        try:
            raw = self._native_controller.run_generations(
                self._start_generation,
                count,
                steps,
                births,
            )
            raw_mapping = _dict_copy(raw, field="native run result")
            if raw_mapping.get("terminal_status") == "failed":
                self._validate_native_manifest(expected_status="failed")
                authoritative = (
                    self._authoritative_terminal_result(
                        expected_status="failed",
                        returned_result=raw_mapping,
                    )
                    if self._persistence_mailbox_required
                    else raw_mapping
                )
                result = self._normalize_failed_result(authoritative)
                self._failure = str(result["root_failure"]["message"])
                self._final = result
                return _detached_copy(result)

            provisional = self._normalize_and_validate_result(
                raw_mapping,
                requested=count,
                expected_terminal_status="awaiting_snapshot_validation",
            )
            if self._phase_e_passive_no_churn_guard is not None:
                fixed_capacity = self._phase_e_passive_no_churn_guard.get(
                    "fixed_capacity"
                )
                if type(fixed_capacity) is not int or fixed_capacity <= 0:
                    raise _fail("Phase E passive modular guard capacity is invalid")
                try:
                    validate_phase_e_passive_modular_no_churn_guard(
                        self._phase_e_passive_no_churn_guard,
                        provisional["final_checkpoint_projection"],
                        completed_generation_count=count,
                        fixed_capacity=fixed_capacity,
                        max_births=births,
                    )
                except Exception as exc:
                    raise _fail(
                        f"Phase E passive modular no-churn validation failed: {exc}"
                    ) from exc
            self._validate_native_manifest(expected_status="awaiting_snapshot_validation")
            provisional_authority_digest = _dict_copy(
                provisional.get("final_checkpoint_projection_digest_manifests"),
                field="provisional final authority digest",
            )
            provisional_source_projection = _dict_copy(
                provisional_authority_digest.get("source_projection"),
                field="provisional final source-projection digest manifest",
            )
            provisional_source_projection_digest = _strict_lowercase_sha256(
                provisional_source_projection.get("digest"),
                field="provisional final source-projection digest",
            )
            final_authority_digest = _projection_digest_manifests(
                provisional["final_checkpoint_projection"]
            )
            if (
                final_authority_digest != provisional_authority_digest
                or provisional["final_checkpoint_projection_digest_manifests"]
                != provisional_authority_digest
            ):
                raise _fail(
                    "final authority digest changed while constructing facade result"
                )
            prepared_receipt: dict[str, Any] | None = None
            if "acknowledgement_settlement_gate_v1" in self._phase4_capabilities:
                prepared_receipt, receipt = (
                    self._settle_snapshot_validation_two_phase(
                        provisional=provisional,
                        source_projection_digest=provisional_source_projection_digest,
                    )
                )
            else:
                receipt = self._normalize_snapshot_validation_receipt(
                    self._native_controller.acknowledge_snapshot_validation(
                        provisional_source_projection_digest
                    ),
                    provisional=provisional,
                )
                self._validate_native_manifest(expected_status="run_complete")
            if self._persistence_mailbox_required:
                authoritative = self._authoritative_terminal_result(
                    expected_status="run_complete"
                )
                result = self._normalize_and_validate_result(
                    authoritative,
                    requested=count,
                    expected_terminal_status="run_complete",
                )
                expected_completed = _detached_copy(provisional)
                expected_completed["terminal_status"] = "run_complete"
                expected_completed["snapshot_frontier"] = expected_completed[
                    "trajectory_commit_frontier"
                ]
                expected_completed["success_marker_allowed"] = True
                if not _exact_values_equal(result, expected_completed):
                    raise _fail(
                        "native authoritative final result changed outside terminal settlement"
                    )
            else:
                result = provisional
                result["terminal_status"] = "run_complete"
                result["snapshot_frontier"] = result["trajectory_commit_frontier"]
                result["success_marker_allowed"] = True
            if prepared_receipt is not None:
                result["snapshot_validation_prepare_receipt"] = prepared_receipt
            result["snapshot_validation_receipt"] = receipt
        except Exception as exc:
            try:
                failed_raw = self._native_controller.fail_snapshot_validation(str(exc))
                self._validate_native_manifest(expected_status="failed")
                failed_mapping = _dict_copy(
                    failed_raw,
                    field="native failed snapshot-validation result",
                )
                failed_authority = (
                    self._authoritative_terminal_result(
                        expected_status="failed",
                        returned_result=failed_mapping,
                    )
                    if self._persistence_mailbox_required
                    else failed_mapping
                )
                result = self._normalize_failed_result(failed_authority)
                self._failure = str(result["root_failure"]["message"])
                self._final = result
                return _detached_copy(result)
            except Exception as failure_exc:
                self._failure = f"{exc}; native failure latch error: {failure_exc}"
                raise _fail(self._failure) from failure_exc

        self._attach_patch_controller_receipt(result)
        self._final = _detached_copy(result)
        return result

    def run_with_persistence(
        self,
        runner: Any,
        generation_count: int,
        *,
        snapshot_schedule: Mapping[str, Any],
        steps_per_generation: int | None = None,
        max_births: int | None = None,
        staging_root: str | Path | None = None,
        poll_timeout_ms: int = 100,
        idle_timeout_seconds: float = 300.0,
        join_timeout_seconds: float = 10.0,
        result_retention: str = "all_boundaries",
        snapshot_observer: Callable[[Mapping[str, Any]], None] | None = None,
        allow_unobserved_terminal_only: bool = False,
    ) -> dict[str, Any]:
        if not self._persistence_mailbox_required:
            raise _fail("Architecture B persistence facade requires mailbox mode")
        if self._trajectory_dispatch_count:
            raise _fail("Architecture B controller is single-use")
        count = _strict_int(generation_count, field="generation_count", minimum=1)
        steps = _strict_int(
            self._default_steps_per_generation
            if steps_per_generation is None
            else steps_per_generation,
            field="steps_per_generation",
            minimum=1,
        )
        if steps != self._default_steps_per_generation:
            raise _fail("steps_per_generation does not match the imported configuration")
        births = _strict_int(
            self._default_max_births if max_births is None else max_births,
            field="max_births",
            minimum=0,
        )
        schedule = _validate_snapshot_schedule(snapshot_schedule)
        if result_retention not in {"all_boundaries", "terminal_only"}:
            raise _fail(
                "result_retention must be 'all_boundaries' or 'terminal_only'"
            )
        if snapshot_observer is not None and not callable(snapshot_observer):
            raise _fail("snapshot_observer must be callable")
        explicit_checkpoints = schedule["explicit_next_generations"]
        target_generation = self._start_generation + count
        if type(allow_unobserved_terminal_only) is not bool:
            raise _fail("allow_unobserved_terminal_only must be boolean")
        if result_retention == "terminal_only" and snapshot_observer is None:
            terminal_only_schedule_exact = (
                allow_unobserved_terminal_only
                and schedule["report_every_generations"] == 0
                and schedule["checkpoint_every_generations"] == 0
                and explicit_checkpoints == [target_generation]
                and schedule["emit_final_snapshot"] is True
            )
            if not terminal_only_schedule_exact:
                raise _fail(
                    "unobserved terminal_only retention requires explicit opt-in, "
                    "disabled periodic reporting/checkpoints, and one terminal checkpoint"
                )
        if snapshot_observer is not None and schedule["report_every_generations"] != 1:
            raise _fail(
                "snapshot_observer requires acknowledgement-protected reporting every generation"
            )
        if any(
            generation <= self._start_generation
            or generation > target_generation
            for generation in explicit_checkpoints
        ):
            raise _fail(
                "snapshot explicit checkpoint generation is outside the requested run"
            )
        final_phase = target_generation - self._schedule_origin_generation
        final_report_due = (
            schedule["report_every_generations"] > 0
            and final_phase % schedule["report_every_generations"] == 0
        )
        final_checkpoint_due = (
            schedule["checkpoint_every_generations"] > 0
            and final_phase % schedule["checkpoint_every_generations"] == 0
        ) or target_generation in explicit_checkpoints
        if (
            self._compact_reporting_only
            and not final_report_due
            and not final_checkpoint_due
        ):
            raise _fail(
                "compact reporting requires a terminal report or checkpoint"
            )
        if (
            isinstance(join_timeout_seconds, bool)
            or not isinstance(join_timeout_seconds, (int, float))
            or not math.isfinite(float(join_timeout_seconds))
            or float(join_timeout_seconds) <= 0
        ):
            raise _fail("join_timeout_seconds must be finite and positive")
        join_timeout = float(join_timeout_seconds)

        outcome: dict[str, Any] = {}
        errors: list[tuple[int, str, BaseException]] = []
        error_lock = threading.Lock()
        error_sequence = 0

        def record_error(source: str, exc: BaseException) -> None:
            nonlocal error_sequence
            with error_lock:
                error_sequence += 1
                errors.append((error_sequence, source, exc))

        def native_dispatch() -> None:
            try:
                arguments = (
                    self._start_generation,
                    count,
                    steps,
                    births,
                    schedule,
                )
                if result_retention == "terminal_only":
                    outcome["raw"] = self._native_controller.run_generations(
                        *arguments,
                        result_retention,
                    )
                else:
                    outcome["raw"] = self._native_controller.run_generations(*arguments)
            except BaseException as exc:
                record_error("native dispatch", exc)

        run_thread = threading.Thread(
            target=native_dispatch,
            name=f"eiel-archb-native-{self._controller_id[:16]}",
            daemon=True,
        )
        try:
            persistence_loop = ArchitectureBPersistenceLoop(
                self._native_controller,
                runner,
                staging_root=staging_root,
                poll_timeout_ms=poll_timeout_ms,
                idle_timeout_seconds=idle_timeout_seconds,
                run_thread=run_thread,
                initial_after_snapshot_sequence=self._initial_boundary_ordinal,
                snapshot_observer=snapshot_observer,
            )
        except BaseException as exc:
            self._failure = f"persistence loop setup: {type(exc).__name__}: {exc}"
            raise _fail(self._failure) from exc

        self._trajectory_dispatch_count = 1
        persistence_result: dict[str, Any] | None = None
        run_thread.start()
        try:
            persistence_result = self._validate_persistence_loop_result(
                persistence_loop.run(),
            )
        except BaseException as exc:
            # The persistence loop is the sole authority that reports its first
            # failure to the native owner. Re-notifying here can replace that
            # root cause or race a terminal native failure result.
            record_error("persistence loop", exc)

        run_thread.join(timeout=join_timeout)
        if run_thread.is_alive():
            timeout_error = TimeoutError(
                "Architecture B native dispatch did not join within the configured bound"
            )
            record_error("native dispatch join", timeout_error)
        if run_thread.is_alive():
            self._failure = "native dispatch thread remained alive after bounded failure join"
            raise _fail(self._failure)

        raw = outcome.get("raw")
        raw_mapping: dict[str, Any] | None = None
        if raw is not None:
            try:
                raw_mapping = _dict_copy(raw, field="native scheduled run result")
            except BaseException as exc:
                record_error("native result", exc)

        if raw_mapping is not None and raw_mapping.get("terminal_status") == "failed":
            try:
                self._validate_native_manifest(expected_status="failed")
                authoritative = self._authoritative_terminal_result(
                    expected_status="failed",
                    returned_result=raw_mapping,
                )
                failed = self._normalize_failed_result(authoritative)
                self._final = _detached_copy(failed)
                native_root = str(failed["root_failure"]["message"])
                self._failure = native_root
                first_error = (
                    min(errors, key=lambda row: row[0])[2]
                    if errors
                    else NativeRunControllerError(native_root)
                )
                raise _fail(native_root) from first_error
            except BaseException as exc:
                if self._final is not None:
                    raise
                record_error("native failed-result validation", exc)

        if errors:
            _, source, first_error = min(errors, key=lambda row: row[0])
            self._failure = f"{source}: {type(first_error).__name__}: {first_error}"
            raise _fail(self._failure) from first_error
        if raw_mapping is None:
            self._failure = "native dispatch returned no result"
            raise _fail(self._failure)
        if persistence_result is None:
            self._failure = "persistence loop returned no result"
            raise _fail(self._failure)
        terminal_status = raw_mapping.get("terminal_status")
        if terminal_status == "cancelled":
            self._validate_native_manifest(expected_status="cancelled")
            authoritative = self._authoritative_terminal_result(
                expected_status="cancelled",
                returned_result=raw_mapping,
            )
            result = self._normalize_and_validate_scheduled_result(
                authoritative,
                requested=count,
                schedule=schedule,
                expected_terminal_status="cancelled",
                result_retention=result_retention,
            )
            if persistence_result.get("terminal_state_seen") != "cancelled":
                raise _fail("persistence loop did not observe native cancellation")
            if persistence_result["last_snapshot_sequence"] != result[
                "final_snapshot_sequence"
            ]:
                raise _fail("persistence loop cancellation snapshot sequence mismatch")
            if persistence_result["checkpoints_persisted"] < 1:
                raise _fail("persistence loop did not persist the cancellation checkpoint")
            result["persistence_loop_result"] = _detached_copy(persistence_result)
            self._attach_patch_controller_receipt(result)
            self._final = _detached_copy(result)
            return result
        if terminal_status != "awaiting_snapshot_validation":
            raise _fail(f"native scheduled run returned invalid terminal status: {terminal_status!r}")
        if (
            persistence_result.get("stop_reason") != "run_thread_finished"
            or persistence_result.get("terminal_state_seen") != "none"
        ):
            raise _fail("persistence loop did not stop at the provisional native result")

        try:
            provisional = self._normalize_and_validate_scheduled_result(
                raw_mapping,
                requested=count,
                schedule=schedule,
                expected_terminal_status="awaiting_snapshot_validation",
                result_retention=result_retention,
            )
            self._validate_native_manifest(
                expected_status="awaiting_snapshot_validation"
            )
            final_projection = _dict_copy(
                provisional.get("final_checkpoint_projection"),
                field="scheduled final checkpoint projection",
            )
            final_manifests = _projection_digest_manifests(final_projection)
            source_manifest = _dict_copy(
                final_manifests.get("source_projection"),
                field="scheduled final source-projection digest manifest",
            )
            source_digest = _strict_lowercase_sha256(
                source_manifest.get("digest"),
                field="scheduled final source-projection digest",
            )
            prepared_receipt, settlement_receipt = (
                self._settle_snapshot_validation_two_phase(
                    provisional=provisional,
                    source_projection_digest=source_digest,
                )
            )
            authoritative = self._authoritative_terminal_result(
                expected_status="run_complete"
            )
            result = self._normalize_and_validate_scheduled_result(
                authoritative,
                requested=count,
                schedule=schedule,
                expected_terminal_status="run_complete",
                result_retention=result_retention,
            )
            expected_completed = _detached_copy(provisional)
            expected_completed["terminal_status"] = "run_complete"
            expected_completed["snapshot_frontier"] = expected_completed[
                "trajectory_commit_frontier"
            ]
            expected_completed["success_marker_allowed"] = True
            if not _exact_values_equal(result, expected_completed):
                raise _fail(
                    "native authoritative final result changed outside terminal settlement"
                )
            result["snapshot_validation_prepare_receipt"] = prepared_receipt
            result["snapshot_validation_receipt"] = settlement_receipt
            result["persistence_loop_result"] = _detached_copy(persistence_result)
            self._attach_patch_controller_receipt(result)
        except BaseException as exc:
            try:
                failed_raw = self._native_controller.fail_snapshot_validation(str(exc))
                self._validate_native_manifest(expected_status="failed")
                failed_mapping = _dict_copy(
                    failed_raw,
                    field="native failed snapshot-validation result",
                )
                failed_authority = self._authoritative_terminal_result(
                    expected_status="failed",
                    returned_result=failed_mapping,
                )
                failed = self._normalize_failed_result(failed_authority)
                self._final = _detached_copy(failed)
            except BaseException as failure_exc:
                self._failure = (
                    f"{exc}; native snapshot-validation failure latch error: "
                    f"{failure_exc}"
                )
                raise _fail(self._failure) from failure_exc
            self._failure = f"scheduled result validation: {type(exc).__name__}: {exc}"
            raise _fail(self._failure) from exc
        self._final = _detached_copy(result)
        return result

    def run_many_with_persistence(
        self,
        generation_count: int,
        *,
        runner: Any,
        snapshot_schedule: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.run_with_persistence(
            runner,
            generation_count,
            snapshot_schedule=snapshot_schedule,
            **kwargs,
        )

    def final_result(self) -> dict[str, Any]:
        if self._final is not None:
            terminal_status = self._final.get("terminal_status")
            manifest = self._native_manifest()
            self._validate_native_identity(
                manifest,
                field="native controller manifest for cached final result",
            )
            if manifest.get("status") != terminal_status:
                raise _fail("cached facade and native terminal statuses disagree")
            if self._persistence_mailbox_required:
                self._authoritative_terminal_result(
                    expected_status=str(terminal_status),
                )
            if terminal_status in {"cancelled", "failed"}:
                _require_bool(self._final, "success_marker_allowed", False)
                for field in (
                    "snapshot_validation_prepare_receipt",
                    "snapshot_validation_receipt",
                    "public_output_finalization",
                    "public_output_finalized",
                ):
                    if field in self._final:
                        raise _fail(
                            f"{terminal_status} result contains success-only field {field}"
                        )
            return _detached_copy(self._final)
        if self._failure:
            raise _fail(f"controller validation is latched failed: {self._failure}")

        outcome = _dict_copy(
            self._native_controller.final_result(),
            field="native final_result",
        )
        if outcome.get("terminal") is False:
            status = _strict_identifier(
                outcome.get("status"),
                field="native final_result status",
            )
            if status in _TERMINAL_STATUSES:
                raise _fail("native final_result marks a terminal status as nonterminal")
            return outcome

        terminal_status = outcome.get("terminal_status")
        if terminal_status not in _TERMINAL_STATUSES:
            raise _fail("native final_result has an invalid terminal status")
        self._validate_native_identity(outcome, field="native final_result")
        self._validate_closed_phase4_result_fields(
            outcome,
            terminal_status=terminal_status,
        )
        if self._persistence_mailbox_required and terminal_status in {
            "cancelled",
            "failed",
        }:
            _require_bool(outcome, "success_marker_allowed", False)
        if self._persistence_mailbox_required and terminal_status == "cancelled":
            _require_bool(outcome, "promotion_result_allowed", False)
            completed = _strict_int(
                outcome.get("completed_generations"),
                field="native final_result completed_generations",
                minimum=1,
            )
            boundaries = outcome.get("boundaries")
            if not isinstance(boundaries, list) or not boundaries:
                raise _fail("native cancelled final_result has no boundary")
            cancelled_boundary = boundaries[-1]
            if not isinstance(cancelled_boundary, Mapping):
                raise _fail("native cancelled final_result boundary is malformed")
            self._validate_cancelled_terminal_receipts(
                outcome,
                cancelled_boundary=cancelled_boundary,
                completed=completed,
            )
        return outcome

    def request_cancel(
        self,
        request_id: str,
        *,
        reason: str = "operator_requested",
    ) -> dict[str, Any]:
        """Request native cancellation at a complete-generation boundary.

        Acceptance only latches intent in the resident owner. It never means
        that the active generation was interrupted or that status is terminal.
        """

        resolved_request_id = _strict_ascii_identifier(
            request_id,
            field="cancel request_id",
        )
        if type(reason) is not str or reason not in _CANCEL_REASONS:
            raise _fail(
                "cancel reason is unsupported; cancellation is available only "
                "at a complete-generation boundary"
            )

        manifest_before = self._native_manifest()
        self._validate_native_identity(
            manifest_before,
            field="native controller manifest before cancellation",
        )
        status_before = _strict_identifier(
            manifest_before.get("status"),
            field="native controller status before cancellation",
        )
        if self._final is not None and self._final.get("terminal_status") != status_before:
            raise _fail("facade and native terminal status disagree before cancellation")

        native_request = {
            "api_version": 1,
            "request_id": resolved_request_id,
            "expected_controller_id": self._controller_id,
            "expected_owner_epoch": self._owner_epoch,
            "reason": reason,
        }
        native_operation = getattr(self._native_controller, "request_cancel", None)
        if not callable(native_operation):
            raise _fail(
                "native owner does not support complete-generation-boundary "
                "cancellation"
            )
        try:
            raw_receipt = native_operation(native_request)
        except (AttributeError, NotImplementedError) as exc:
            raise _fail(
                "native owner does not support complete-generation-boundary "
                "cancellation"
            ) from exc
        except Exception as exc:
            raise _fail(f"native cancellation request failed: {exc}") from exc

        manifest_after = self._native_manifest()
        self._validate_native_identity(
            manifest_after,
            field="native controller manifest after cancellation",
        )
        return self._normalize_cancel_receipt(
            raw_receipt,
            request_id=resolved_request_id,
            reason=reason,
            manifest_before=manifest_before,
            manifest_after=manifest_after,
        )

    def manifest(self) -> dict[str, Any]:
        native_manifest = self._native_manifest()
        self._validate_native_identity(
            native_manifest,
            field="native controller manifest",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed" if self._failure else str(native_manifest.get("status", "")),
            "initial_import_count": 1,
            "trajectory_dispatch_count": int(self._trajectory_dispatch_count),
            "python_generation_dispatch_count": 0,
            "python_callback_invocation_count": 0,
            "runtime_jit_request_count": 0,
            "hidden_fallback_count": 0,
            "boundary_tensor_manifest_bytes": int(self._boundary_tensor_manifest_bytes),
            "extension_path": self._extension_path,
            "run_id": self._run_id,
            "controller_id": self._controller_id,
            "owner_epoch": self._owner_epoch,
            "persistence_mailbox_required": self._persistence_mailbox_required,
            "initial_boundary_ordinal": self._initial_boundary_ordinal,
            "schedule_origin_generation": self._schedule_origin_generation,
            "phase4_capabilities": sorted(self._phase4_capabilities),
            "safe_default_decision": _detached_copy(self._safe_default_decision),
            "native_controller": native_manifest,
            "failure": self._failure,
        }
        if self._architecture_world == "patch":
            manifest.update(
                {
                    "world": "patch",
                    "native_runtime_patch_profile": self._patch_profile,
                    "resident_composer": (
                        "execute_generation_patch_with_population_birth_commit_controlled_cpu"
                    ),
                    "controller_capability": (
                        NATIVE_PATCH_CONTROLLED_COMPOSER_CAPABILITY
                    ),
                    "initial_owner_count": 1,
                    "controller_call_count": int(self._trajectory_dispatch_count),
                    "python_tick_callback_invocation_count": 0,
                    "python_generation_trajectory_callback_invocation_count": 0,
                    "production_routing_enabled": False,
                    "safe_default_promoted": False,
                    "promotion_accepted": False,
                    "cuda_supported": False,
                    "instrumentation_enabled": self._patch_instrumentation_enabled,
                    "instrumentation_schema_version": (
                        NATIVE_PATCH_INSTRUMENTATION_RECEIPT_SCHEMA_VERSION
                    ),
                    "patch_extension_capabilities": _detached_copy(
                        self._patch_extension_capabilities
                    ),
                }
            )
        return manifest

    def _attach_patch_controller_receipt(self, result: dict[str, Any]) -> None:
        if self._architecture_world != "patch":
            return
        _require_bool(result, "hidden_fallback", False)
        _require_bool(result, "runtime_jit", False)
        _require_bool(result, "promotion_result_allowed", False)
        counters = {
            "controller_call_count": _strict_int(
                result.get("controller_invocations"),
                field="Patch controller_invocations",
                minimum=0,
            ),
            "native_generation_count": _strict_int(
                result.get("native_generation_transactions"),
                field="Patch native_generation_transactions",
                minimum=0,
            ),
            "python_generation_trajectory_callback_count": _strict_int(
                result.get("python_generation_dispatches"),
                field="Patch python_generation_dispatches",
                minimum=0,
            ),
            "boundary_snapshot_count": _strict_int(
                result.get("boundary_snapshot_count"),
                field="Patch boundary_snapshot_count",
                minimum=0,
            ),
            "boundary_projection_export_count": _strict_int(
                result.get("projection_export_count"),
                field="Patch projection_export_count",
                minimum=0,
            ),
        }
        if counters["controller_call_count"] != 1:
            raise _fail("Patch Architecture B must use exactly one controller call")
        if self._trajectory_dispatch_count != counters["controller_call_count"]:
            raise _fail(
                "Patch Architecture B controller call count contradicts facade dispatch"
            )
        if counters["python_generation_trajectory_callback_count"] != 0:
            raise _fail("Patch Architecture B crossed a Python generation callback")
        native_manifest = self._native_manifest()
        if _strict_int(
            native_manifest.get("python_callback_entry_points"),
            field="Patch native python_callback_entry_points",
            minimum=0,
        ) != 0:
            raise _fail("Patch Architecture B exposed a Python tick callback")
        result["architecture_b_patch_controller_receipt"] = {
            "schema_version": NATIVE_PATCH_CONTROLLER_RECEIPT_SCHEMA_VERSION,
            "world": "patch",
            "profile": self._patch_profile,
            "resident_composer": (
                "execute_generation_patch_with_population_birth_commit_controlled_cpu"
            ),
            "controller_capability": NATIVE_PATCH_CONTROLLED_COMPOSER_CAPABILITY,
            "run_id": self._run_id,
            "controller_id": self._controller_id,
            "owner_epoch": self._owner_epoch,
            "initial_import_count": 1,
            "initial_owner_count": 1,
            **counters,
            "python_tick_callback_count": 0,
            "inter_tick_control_window": "native_declared_cancellation_only",
            "hidden_fallback_count": 0,
            "runtime_compilation_count": 0,
            "production_routing_enabled": False,
            "safe_default_promoted": False,
            "promotion_accepted": False,
            "cuda_supported": False,
        }
        self._attach_patch_instrumentation_receipt(result)

    def _extract_patch_native_instrumentation(
        self,
        result: dict[str, Any],
    ) -> None:
        if self._architecture_world != "patch":
            return
        raw = result.pop(NATIVE_PATCH_INSTRUMENTATION_RESULT_FIELD, _MISSING)
        if raw is _MISSING:
            return
        if not self._patch_instrumentation_enabled:
            raise _fail("native Patch instrumentation was emitted while disabled")
        payload = _dict_copy(raw, field="native Patch instrumentation")
        actual_fields = frozenset(payload)
        allowed_fields = _PATCH_NATIVE_INSTRUMENTATION_FIELDS | frozenset(
            {"schema_version"}
        )
        missing = sorted({"schema_version"} - actual_fields)
        unexpected = sorted(actual_fields - allowed_fields)
        if missing or unexpected:
            raise _fail(
                "native Patch instrumentation fields mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        if payload["schema_version"] != NATIVE_PATCH_INSTRUMENTATION_SCHEMA_VERSION:
            raise _fail("native Patch instrumentation schema mismatch")
        normalized = {
            field: _strict_int(
                payload[field],
                field=f"native Patch instrumentation.{field}",
                minimum=0,
            )
            for field in _PATCH_NATIVE_INSTRUMENTATION_FIELDS
            if field in payload
        }
        for field in (
            "device_transfer_time_ns",
            "device_synchronization_time_ns",
        ):
            if normalized.get(field, 0) != 0:
                raise _fail(
                    f"native Patch instrumentation.{field} must be zero on CPU"
                )
        if (
            self._patch_native_instrumentation is not None
            and normalized != self._patch_native_instrumentation
        ):
            raise _fail("native Patch instrumentation changed during settlement")
        self._patch_native_instrumentation = normalized

    def _attach_patch_instrumentation_receipt(
        self,
        result: dict[str, Any],
    ) -> None:
        measurements: dict[str, int | None] = {
            field: None for field in _PATCH_INSTRUMENTATION_MEASUREMENT_FIELDS
        }
        sources = {
            field: (
                "disabled" if not self._patch_instrumentation_enabled else "not_reported"
            )
            for field in _PATCH_INSTRUMENTATION_MEASUREMENT_FIELDS
        }
        measurements["device_transfer_time_ns"] = 0
        measurements["device_synchronization_time_ns"] = 0
        sources["device_transfer_time_ns"] = "not_applicable_cpu"
        sources["device_synchronization_time_ns"] = "not_applicable_cpu"
        initial_breakdown: dict[str, int | None] = {
            "unique_tensor_storage_bytes": None,
            "unique_tensor_storage_count": None,
        }

        native = self._patch_native_instrumentation or {}
        if self._patch_instrumentation_enabled:
            if self._patch_initial_import_metrics is None:
                raise _fail("enabled Patch instrumentation lost initial import metrics")
            measurements["initial_import_bytes"] = self._patch_initial_import_metrics[
                "represented_payload_bytes"
            ]
            sources["initial_import_bytes"] = "facade_import_crossing_metrics"
            initial_breakdown = {
                "unique_tensor_storage_bytes": self._patch_initial_import_metrics[
                    "unique_tensor_storage_bytes"
                ],
                "unique_tensor_storage_count": self._patch_initial_import_metrics[
                    "unique_tensor_storage_count"
                ],
            }

            boundary_bytes = _strict_int(
                result.get("boundary_tensor_manifest_bytes"),
                field="Patch instrumentation boundary_tensor_manifest_bytes",
                minimum=0,
            )
            measurements["generation_boundary_projection_bytes"] = boundary_bytes
            sources["generation_boundary_projection_bytes"] = (
                "validated_boundary_tensor_manifests"
            )
            boundaries = result.get("boundaries")
            if not isinstance(boundaries, list):
                raise _fail("Patch instrumentation result boundaries are not a list")
            checkpoint_bytes = 0
            for index, boundary in enumerate(boundaries):
                if not isinstance(boundary, Mapping):
                    raise _fail(
                        f"Patch instrumentation boundaries[{index}] is not a mapping"
                    )
                byte_manifest = boundary.get("tensor_manifest_boundary_bytes")
                if not isinstance(byte_manifest, Mapping):
                    raise _fail(
                        "Patch instrumentation boundary has no validated byte manifest"
                    )
                checkpoint_bytes += _strict_int(
                    byte_manifest.get("post_turnover_checkpoint_projection"),
                    field=(
                        "Patch instrumentation checkpoint projection bytes at "
                        f"boundary {index}"
                    ),
                    minimum=0,
                )
            measurements["checkpoint_bytes"] = checkpoint_bytes
            sources["checkpoint_bytes"] = "validated_boundary_tensor_manifests"

            for field, value in native.items():
                measurements[field] = value
                sources[field] = "native_instrumentation"
            boundary_wall_time = result.get("boundary_wall_time_ns", _MISSING)
            if boundary_wall_time is not _MISSING:
                measurements["controller_boundary_wall_time_ns"] = _strict_int(
                    boundary_wall_time,
                    field="Patch instrumentation boundary_wall_time_ns",
                    minimum=0,
                )
                sources["controller_boundary_wall_time_ns"] = (
                    "validated_native_result"
                )
            if (
                "host_projection_time_ns" not in native
                and "checkpoint_projection_time_ns" in native
                and "reporting_projection_time_ns" in native
            ):
                measurements["host_projection_time_ns"] = (
                    native["checkpoint_projection_time_ns"]
                    + native["reporting_projection_time_ns"]
                )
                sources["host_projection_time_ns"] = "derived_native_projection_times"

        result["architecture_b_patch_instrumentation_receipt"] = {
            "schema_version": NATIVE_PATCH_INSTRUMENTATION_RECEIPT_SCHEMA_VERSION,
            "enabled": self._patch_instrumentation_enabled,
            "status": "enabled" if self._patch_instrumentation_enabled else "disabled",
            "observational_only": True,
            "world": "patch",
            "device": "cpu",
            "device_transfer_applicability": "not_applicable_cpu",
            "performance_threshold_applied": False,
            "measurement_units": dict(_PATCH_INSTRUMENTATION_UNITS),
            "measurements": measurements,
            "measurement_sources": sources,
            "initial_import_breakdown": initial_breakdown,
            "native_fields_present": sorted(native),
        }

    def _native_manifest(self) -> dict[str, Any]:
        return _dict_copy(
            self._native_controller.controller_manifest(),
            field="native controller_manifest",
        )

    def _validate_closed_phase4_result_fields(
        self,
        result: dict[str, Any],
        *,
        terminal_status: str,
    ) -> None:
        if not self._persistence_mailbox_required:
            return
        if terminal_status == "failed":
            expected_fields = _PHASE4_FAILED_RESULT_FIELDS
        elif terminal_status == "cancelled":
            expected_fields = _PHASE4_CANCELLED_RESULT_FIELDS
        elif terminal_status in {
            "awaiting_snapshot_validation",
            "run_complete",
        }:
            expected_fields = _PHASE4_RUN_RESULT_FIELDS
        else:
            raise _fail(f"unsupported Phase 4 result status: {terminal_status!r}")
        _closed_dict(
            result,
            field=f"native Phase 4 {terminal_status} result",
            required_fields=expected_fields,
        )

    def _authoritative_terminal_result(
        self,
        *,
        expected_status: str,
        returned_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = _dict_copy(
            self._native_controller.final_result(),
            field="native authoritative final_result",
        )
        if outcome.get("terminal") is False:
            raise _fail("native final_result is nonterminal after terminal publication")
        if outcome.get("terminal_status") != expected_status:
            raise _fail(
                "native authoritative terminal status mismatch: "
                f"expected={expected_status!r} "
                f"actual={outcome.get('terminal_status')!r}"
            )
        self._validate_native_identity(
            outcome,
            field="native authoritative final_result",
        )
        if returned_result is not None and not _exact_values_equal(
            outcome,
            returned_result,
        ):
            raise _fail("native returned result differs from authoritative final_result")
        return outcome

    def _validate_persistence_loop_result(
        self,
        raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _closed_dict(
            _dict_copy(raw, field="Architecture B persistence loop result"),
            field="Architecture B persistence loop result",
            required_fields=_PERSISTENCE_LOOP_RESULT_FIELDS,
        )
        if result["schema_version"] != (
            "eiel_architecture_b_persistence_loop_result_v1"
        ):
            raise _fail("persistence loop result schema mismatch")
        for field in (
            "snapshots_seen",
            "progress_only_snapshots",
            "skipped_snapshot_sequences",
            "persistence_acknowledgements",
            "identical_acknowledgement_retries",
            "checkpoints_persisted",
            "last_snapshot_sequence",
        ):
            _strict_int(
                result[field],
                field=f"persistence_loop_result.{field}",
                minimum=0,
            )
        paths = result["architecture_b_checkpoint_bundle_paths"]
        if type(paths) is not list:
            raise _fail(
                "persistence_loop_result.architecture_b_checkpoint_bundle_paths "
                "must be an array"
            )
        for index, path in enumerate(paths):
            _strict_identifier(
                path,
                field=(
                    "persistence_loop_result.architecture_b_checkpoint_bundle_paths"
                    f"[{index}]"
                ),
            )
        if len(paths) != result["checkpoints_persisted"]:
            raise _fail("persistence loop checkpoint path count mismatch")
        return result

    def _normalize_cancel_receipt(
        self,
        raw: Any,
        *,
        request_id: str,
        reason: str,
        manifest_before: Mapping[str, Any],
        manifest_after: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = _dict_copy(raw, field="native cancellation receipt")
        actual_fields = frozenset(receipt)
        if actual_fields != _CANCEL_RECEIPT_FIELDS:
            missing = sorted(_CANCEL_RECEIPT_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _CANCEL_RECEIPT_FIELDS)
            raise _fail(
                "native cancellation receipt fields mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        if receipt.get("request_id") != request_id:
            raise _fail("native cancellation receipt request identity mismatch")
        if receipt.get("controller_id") != self._controller_id:
            raise _fail("native cancellation receipt controller identity mismatch")
        receipt_owner_epoch = _strict_int(
            receipt.get("owner_epoch"),
            field="native cancellation receipt owner_epoch",
            minimum=1,
        )
        if receipt_owner_epoch != self._owner_epoch:
            raise _fail("native cancellation receipt owner epoch mismatch")
        _require_bool(receipt, "terminal_status_changed", False)

        disposition = receipt.get("disposition")
        if disposition not in _CANCEL_DISPOSITIONS:
            raise _fail("native cancellation receipt has an invalid disposition")
        if disposition == "identity_mismatch":
            raise _fail("native owner rejected its facade cancellation identity")

        state_observed = _strict_identifier(
            receipt.get("state_observed"),
            field="native cancellation receipt state_observed",
        )
        first_request_id = receipt.get("first_cancel_request_id")
        cancel_sequence = receipt.get("cancel_sequence")
        if first_request_id is not None:
            first_request_id = _strict_ascii_identifier(
                first_request_id,
                field="native cancellation receipt first_cancel_request_id",
            )
        if cancel_sequence is not None:
            cancel_sequence = _strict_int(
                cancel_sequence,
                field="native cancellation receipt cancel_sequence",
                minimum=1,
            )
        if (first_request_id is None) != (cancel_sequence is None):
            raise _fail(
                "native cancellation receipt first request and sequence disagree"
            )
        if cancel_sequence is not None and cancel_sequence != 1:
            raise _fail("native cancellation receipt sequence is not the first latch")

        status_before = _strict_identifier(
            manifest_before.get("status"),
            field="native controller status before cancellation",
        )
        status_after = _strict_identifier(
            manifest_after.get("status"),
            field="native controller status after cancellation",
        )

        if disposition == "accepted":
            if (
                state_observed != "running"
                or first_request_id != request_id
                or cancel_sequence is None
            ):
                raise _fail("native cancellation acceptance is inconsistent")
            if status_before in _TERMINAL_STATUSES:
                raise _fail("native accepted cancellation outside an active run")
            if status_after == "run_complete" or status_after == "initialized":
                raise _fail("native accepted cancellation has an invalid post-state")
            if self._accepted_cancel_identity is not None:
                raise _fail("native accepted more than one cancellation latch")
            self._accepted_cancel_identity = (first_request_id, cancel_sequence)
            self._accepted_cancel_reason = reason
        elif disposition == "already_requested":
            if (
                state_observed != "running"
                or first_request_id is None
                or cancel_sequence is None
            ):
                raise _fail("native repeated cancellation receipt is inconsistent")
            if status_before in _TERMINAL_STATUSES:
                raise _fail("native repeated cancellation is outside an active run")
            if status_after == "run_complete" or status_after == "initialized":
                raise _fail("native repeated cancellation has an invalid post-state")
            if self._accepted_cancel_identity is None:
                self._accepted_cancel_identity = (first_request_id, cancel_sequence)
            elif self._accepted_cancel_identity != (first_request_id, cancel_sequence):
                raise _fail("native repeated cancellation changed the first request")
        elif disposition == "already_terminal":
            if state_observed not in _TERMINAL_STATUSES:
                raise _fail("native terminal cancellation receipt is inconsistent")
            if status_before in _TERMINAL_STATUSES and state_observed != status_before:
                raise _fail(
                    "native cancellation changed the facade terminal status"
                )
            if status_after != state_observed:
                raise _fail("native terminal cancellation receipt disagrees with post-state")
            if state_observed == "cancelled" and first_request_id is None:
                raise _fail("cancelled native owner has no accepted cancellation")
            if first_request_id is not None:
                observed_identity = (first_request_id, cancel_sequence)
                if self._accepted_cancel_identity is None:
                    self._accepted_cancel_identity = observed_identity
                elif self._accepted_cancel_identity != observed_identity:
                    raise _fail("native terminal cancellation changed the first request")
        elif disposition == "not_running":
            if (
                state_observed != "initialized"
                or first_request_id is not None
                or cancel_sequence is not None
            ):
                raise _fail("native non-running cancellation receipt is inconsistent")
            if status_before != "initialized" or status_after != "initialized":
                raise _fail("native non-running cancellation changed controller state")
        elif disposition == "control_point_unavailable":
            if (
                state_observed != "running"
                or first_request_id is not None
                or cancel_sequence is not None
            ):
                raise _fail(
                    "native unavailable-control-point cancellation receipt is inconsistent"
                )
            if status_before in _TERMINAL_STATUSES or status_before == "initialized":
                raise _fail(
                    "native unavailable-control-point cancellation is outside an active run"
                )
            if status_after == "initialized":
                raise _fail(
                    "native unavailable-control-point cancellation reset controller state"
                )

        if status_before in _TERMINAL_STATUSES and disposition != "already_terminal":
            raise _fail("native cancellation did not preserve terminal status")
        receipt["first_cancel_request_id"] = first_request_id
        receipt["cancel_sequence"] = cancel_sequence
        return receipt

    def _validate_cancelled_terminal_receipts(
        self,
        result: dict[str, Any],
        *,
        cancelled_boundary: Mapping[str, Any],
        completed: int,
    ) -> None:
        request_id = _strict_ascii_identifier(
            result.get("cancel_request_id"),
            field="cancelled run cancel_request_id",
        )
        safe_generation = self._start_generation + completed
        trajectory_frontier = safe_generation - 1
        snapshot_sequence = self._initial_boundary_ordinal + completed
        snapshot_id = (
            f"{self._controller_id}:owner:{self._owner_epoch}:"
            f"boundary:{snapshot_sequence}"
        )
        safe_cursor = f"generation:{safe_generation}:tick:0"

        if result.get("cancel_request_observation_cursor") != safe_cursor:
            raise _fail("cancelled run request observation cursor mismatch")
        if result.get("final_safe_cursor") != safe_cursor:
            raise _fail("cancelled run final safe cursor mismatch")
        if _strict_int(
            result.get("output_ack_frontier"),
            field="cancelled run output_ack_frontier",
            minimum=-1,
        ) != trajectory_frontier:
            raise _fail("cancelled run output acknowledgement frontier mismatch")
        _require_bool(result, "cancellation_checkpoint_requested", True)
        _require_bool(result, "cancellation_checkpoint_persisted", True)
        _require_bool(result, "cancelled_snapshot_durable", True)

        request_receipt = _closed_dict(
            _dict_copy(
                result.get("cancel_request_receipt"),
                field="native cancellation request receipt",
            ),
            field="native cancellation request receipt",
            required_fields=_CANCEL_RECEIPT_FIELDS,
        )
        request_receipt_expected = {
            "request_id": request_id,
            "controller_id": self._controller_id,
            "owner_epoch": self._owner_epoch,
            "disposition": "accepted",
            "first_cancel_request_id": request_id,
            "cancel_sequence": 1,
            "state_observed": "running",
            "terminal_status_changed": False,
        }
        if request_receipt != request_receipt_expected:
            raise _fail("native cancellation request receipt values mismatch")
        accepted_identity = (request_id, 1)
        if self._accepted_cancel_identity is None:
            self._accepted_cancel_identity = accepted_identity
        elif self._accepted_cancel_identity != accepted_identity:
            raise _fail("cancelled result changed the first accepted request")

        observation = _closed_dict(
            _dict_copy(
                result.get("cancel_observation_receipt"),
                field="native cancellation observation receipt",
            ),
            field="native cancellation observation receipt",
            required_fields=_CANCEL_OBSERVATION_RECEIPT_FIELDS,
        )
        if observation.get("schema_version") != (
            "eiel_native_architecture_b_cancellation_observation_receipt_v1"
        ):
            raise _fail("cancellation observation receipt schema mismatch")
        if observation.get("request_id") != request_id:
            raise _fail("cancellation observation receipt request identity mismatch")
        observation_reason = observation.get("reason")
        if observation_reason not in _CANCEL_REASONS:
            raise _fail("cancellation observation receipt reason is unsupported")
        if self._accepted_cancel_reason is None:
            self._accepted_cancel_reason = str(observation_reason)
        elif observation_reason != self._accepted_cancel_reason:
            raise _fail("cancellation observation receipt changed the accepted reason")
        if observation.get("controller_id") != self._controller_id:
            raise _fail("cancellation observation receipt controller identity mismatch")
        if _strict_int(
            observation.get("owner_epoch"),
            field="cancellation observation receipt owner_epoch",
            minimum=1,
        ) != self._owner_epoch:
            raise _fail("cancellation observation receipt owner epoch mismatch")
        if _strict_int(
            observation.get("cancel_sequence"),
            field="cancellation observation receipt cancel_sequence",
            minimum=1,
        ) != 1:
            raise _fail("cancellation observation receipt sequence mismatch")
        if observation.get("observation_cursor") != safe_cursor:
            raise _fail("cancellation observation receipt cursor mismatch")
        observation_scalars = {
            "observed_after_completed_generation": trajectory_frontier,
            "boundary_ordinal": snapshot_sequence,
            "snapshot_sequence": snapshot_sequence,
        }
        for field, expected in observation_scalars.items():
            if _strict_int(
                observation.get(field),
                field=f"cancellation observation receipt {field}",
                minimum=0,
            ) != expected:
                raise _fail(f"cancellation observation receipt {field} mismatch")
        if observation.get("snapshot_id") != snapshot_id:
            raise _fail("cancellation observation receipt snapshot identity mismatch")
        snapshot_digest = _strict_lowercase_sha256(
            observation.get("snapshot_digest_sha256"),
            field="cancellation observation receipt snapshot digest",
        )
        boundary_digest = cancelled_boundary.get("snapshot_digest_sha256")
        if boundary_digest is not None and boundary_digest != snapshot_digest:
            raise _fail("cancellation observation receipt snapshot digest mismatch")
        _require_bool(observation, "checkpoint_requested", True)
        _require_bool(observation, "checkpoint_persisted", True)

        terminal = _closed_dict(
            _dict_copy(
                result.get("cancel_terminal_receipt"),
                field="native cancellation terminal receipt",
            ),
            field="native cancellation terminal receipt",
            required_fields=_CANCEL_TERMINAL_RECEIPT_FIELDS,
        )
        terminal_expected_values = {
            "schema_version": (
                "eiel_native_architecture_b_cancellation_terminal_receipt_v1"
            ),
            "request_id": request_id,
            "controller_id": self._controller_id,
            "owner_epoch": self._owner_epoch,
            "terminal_status": "cancelled",
            "request_observation_cursor": safe_cursor,
            "final_safe_cursor": safe_cursor,
            "trajectory_commit_frontier": trajectory_frontier,
            "snapshot_frontier": trajectory_frontier,
            "output_ack_frontier": trajectory_frontier,
            "snapshot_sequence": snapshot_sequence,
            "snapshot_id": snapshot_id,
            "snapshot_digest_sha256": snapshot_digest,
            "cancellation_checkpoint_requested": True,
            "cancellation_checkpoint_persisted": True,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        for field, expected in terminal_expected_values.items():
            if terminal.get(field) != expected or type(terminal.get(field)) is not type(
                expected
            ):
                label = {
                    "cancellation_checkpoint_requested": "checkpoint request",
                    "cancellation_checkpoint_persisted": "checkpoint persistence",
                }.get(field, field.replace("_", " "))
                raise _fail(f"cancellation terminal receipt {label} mismatch")

        result["cancel_request_receipt"] = request_receipt
        result["cancel_observation_receipt"] = observation
        result["cancel_terminal_receipt"] = terminal

    def _validate_native_identity(
        self,
        mapping: Mapping[str, Any],
        *,
        field: str,
    ) -> None:
        if mapping.get("run_id") != self._run_id:
            raise _fail(f"{field} run identity mismatch")
        if mapping.get("controller_id") != self._controller_id:
            raise _fail(f"{field} controller identity mismatch")
        owner_epoch = _strict_int(
            mapping.get("owner_epoch"),
            field=f"{field}.owner_epoch",
            minimum=1,
        )
        if owner_epoch != self._owner_epoch:
            raise _fail(f"{field} owner epoch mismatch")

    def _validate_native_manifest(
        self,
        *,
        expected_status: str,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = (
            self._native_manifest()
            if manifest is None
            else _dict_copy(manifest, field="native controller_manifest")
        )
        if normalized.get("schema_version") != "eiel_native_architecture_b_controller_manifest_v1":
            raise _fail("native controller manifest schema mismatch")
        if normalized.get("status") != expected_status:
            raise _fail(
                "native controller status mismatch: "
                f"expected={expected_status} actual={normalized.get('status')}"
            )
        self._validate_native_identity(
            normalized,
            field="native controller manifest",
        )
        run_started = expected_status != "initialized"
        sealed = expected_status in {"run_complete", "failed", "cancelled"}
        _require_bool(normalized, "run_started", run_started)
        _require_bool(normalized, "sealed", sealed)
        _require_bool(normalized, "native_owner_private", True)
        if (
            self._persistence_mailbox_required
            or "persistence_mailbox_required" in normalized
        ):
            _require_bool(
                normalized,
                "persistence_mailbox_required",
                self._persistence_mailbox_required,
            )
        if self._persistence_mailbox_required:
            if normalized.get("controller_protocol_version") != (
                NATIVE_CONTROLLER_PROTOCOL_VERSION
            ):
                raise _fail("native controller Phase 4 protocol version mismatch")
            if normalized.get("controller_protocol_schema_version") != (
                NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION
            ):
                raise _fail("native controller Phase 4 protocol schema mismatch")
            native_capabilities = normalized.get("controller_protocol_capabilities")
            if not isinstance(native_capabilities, Mapping):
                raise _fail("native controller Phase 4 capability manifest is missing")
            missing_capabilities = sorted(
                name
                for name in NATIVE_PHASE4_REQUIRED_CAPABILITIES
                if native_capabilities.get(name) is not True
            )
            if missing_capabilities:
                raise _fail(
                    "native controller Phase 4 capability manifest is incomplete: "
                    f"{missing_capabilities}"
                )
        if _strict_int(
            normalized.get("python_callback_entry_points"),
            field="python_callback_entry_points",
            minimum=0,
        ) != 0:
            raise _fail("native controller exposes Python callback entry points")

    def _scheduled_boundary_expectations(
        self,
        *,
        completed: int,
        requested: int,
        schedule: Mapping[str, Any],
        terminal_cancelled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        progress_cadence = int(schedule["progress_every_generations"])
        report_cadence = int(schedule["report_every_generations"])
        checkpoint_cadence = int(schedule["checkpoint_every_generations"])
        explicit_checkpoints = frozenset(schedule["explicit_next_generations"])
        expectations: list[dict[str, Any]] = []
        projection_exports = 0
        for offset in range(completed):
            generation = self._start_generation + offset
            phase = generation + 1 - self._schedule_origin_generation
            planned_final = offset + 1 == requested
            cancellation_boundary = terminal_cancelled and offset + 1 == completed
            progress_due = (
                progress_cadence > 0 and phase % progress_cadence == 0
            ) or planned_final
            report_due = (
                report_cadence > 0 and phase % report_cadence == 0
            )
            checkpoint_due = (
                checkpoint_cadence > 0 and phase % checkpoint_cadence == 0
            ) or generation + 1 in explicit_checkpoints or (
                cancellation_boundary and bool(schedule.get("checkpoint_on_cancel", False))
            )
            exact_due = report_due or (
                planned_final and not self._compact_reporting_only
            )
            projection_exports += (2 if exact_due else 0) + (
                1 if checkpoint_due and not exact_due else 0
            )
            if not (progress_due or exact_due or checkpoint_due):
                continue
            expectations.append(
                {
                    "offset": offset,
                    "generation": generation,
                    "snapshot_sequence": self._initial_boundary_ordinal + offset + 1,
                    "progress_due": progress_due,
                    "exact_due": exact_due,
                    "checkpoint_due": checkpoint_due,
                    "planned_final": planned_final,
                    "cancellation_boundary": cancellation_boundary,
                    "persistence_ack_required": exact_due or checkpoint_due,
                }
            )
        return expectations, projection_exports

    def _normalize_scheduled_boundary(
        self,
        raw: Any,
        *,
        expected: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> dict[str, Any]:
        generation = int(expected["generation"])
        sequence = int(expected["snapshot_sequence"])
        acknowledgement_required = bool(expected["persistence_ack_required"])
        boundary = _dict_copy(raw, field=f"scheduled boundary {sequence}")
        if boundary.get("schema_version") != NATIVE_BOUNDARY_SCHEMA_VERSION:
            raise _fail(f"scheduled boundary {sequence} schema mismatch")
        self._validate_native_identity(
            boundary, field=f"scheduled boundary {sequence}"
        )
        expected_scalars = {
            "snapshot_sequence": sequence,
            "boundary_ordinal": sequence,
            "completed_generation": generation,
            "next_generation": generation + 1,
            "tick": 0,
        }
        for field, expected_value in expected_scalars.items():
            if _strict_int(
                boundary.get(field),
                field=f"scheduled boundary {sequence}.{field}",
                minimum=0,
            ) != expected_value:
                raise _fail(f"scheduled boundary {sequence}.{field} mismatch")
        if boundary.get("phase") != "generation_complete":
            raise _fail(f"scheduled boundary {sequence} has an invalid phase")
        expected_snapshot_id = (
            f"{self._controller_id}:owner:{self._owner_epoch}:boundary:{sequence}"
        )
        if boundary.get("snapshot_id") != expected_snapshot_id:
            raise _fail(f"scheduled boundary {sequence} snapshot identity mismatch")
        expected_clone_status = (
            "generation_complete_waiting_persistence"
            if acknowledgement_required
            else "generation_complete_ready_to_resolve_terminal"
        )
        if boundary.get("controller_status_at_clone") != expected_clone_status:
            raise _fail(f"scheduled boundary {sequence} clone status mismatch")

        due_mask = _closed_dict(
            boundary.get("due_mask"),
            field=f"scheduled boundary {sequence}.due_mask",
            required_fields=frozenset(
                {"progress", "exact_reporting", "checkpoint", "final"}
            ),
        )
        expected_due_mask = {
            "progress": bool(expected["progress_due"]),
            "exact_reporting": bool(expected["exact_due"]),
            "checkpoint": bool(expected["checkpoint_due"]),
            "final": bool(expected["planned_final"]),
        }
        if due_mask != expected_due_mask:
            raise _fail(f"scheduled boundary {sequence} due mask mismatch")

        schedule_state_fields = frozenset(
            {
                "schema_version",
                "schedule_origin_generation",
                "imported_boundary_ordinal",
                "progress_every_generations",
                "report_every_generations",
                "checkpoint_every_generations",
                "checkpoint_on_cancel",
                "explicit_next_generations",
            }
        )
        schedule_state = _closed_dict(
            boundary.get("schedule_state"),
            field=f"scheduled boundary {sequence}.schedule_state",
            required_fields=schedule_state_fields,
        )
        expected_schedule_state = {
            "schema_version": NATIVE_SNAPSHOT_SCHEDULE_SCHEMA_VERSION,
            "schedule_origin_generation": self._schedule_origin_generation,
            "imported_boundary_ordinal": self._initial_boundary_ordinal,
            "progress_every_generations": schedule["progress_every_generations"],
            "report_every_generations": schedule["report_every_generations"],
            "checkpoint_every_generations": schedule[
                "checkpoint_every_generations"
            ],
            "checkpoint_on_cancel": schedule["checkpoint_on_cancel"],
            "explicit_next_generations": schedule[
                "explicit_next_generations"
            ],
        }
        if schedule_state != expected_schedule_state:
            raise _fail(f"scheduled boundary {sequence} schedule state mismatch")

        cancellation_boundary = bool(expected["cancellation_boundary"])
        _require_bool(
            boundary,
            "cancellation_observed_at_snapshot_cutoff",
            cancellation_boundary,
        )
        _require_bool(
            boundary,
            "cancellation_checkpoint_requested",
            cancellation_boundary,
        )

        compact_progress = _dict_copy(
            boundary.get("compact_progress"),
            field=f"scheduled boundary {sequence}.compact_progress",
        )
        for field, expected_value in (
            ("completed_generation", generation),
            ("next_generation", generation + 1),
            ("tick", 0),
        ):
            if _strict_int(
                compact_progress.get(field),
                field=f"scheduled boundary {sequence}.compact_progress.{field}",
                minimum=0,
            ) != expected_value:
                raise _fail(
                    f"scheduled boundary {sequence} compact progress mismatch"
                )
        _strict_int(
            compact_progress.get("alive_count"),
            field=f"scheduled boundary {sequence}.compact_progress.alive_count",
            minimum=0,
        )

        _require_bool(
            boundary,
            "persistence_ack_required",
            acknowledgement_required,
        )
        _require_bool(boundary, "snapshot_digest_pending_adapter", False)
        snapshot_digest = _strict_lowercase_sha256(
            boundary.get("snapshot_digest_sha256"),
            field=f"scheduled boundary {sequence}.snapshot_digest_sha256",
        )
        if boundary.get("snapshot_digest") != snapshot_digest:
            raise _fail(f"scheduled boundary {sequence} snapshot digest mismatch")

        exact_due = bool(expected["exact_due"])
        checkpoint_due = bool(expected["checkpoint_due"])
        expected_profile = (
            "canonical_checkpoint"
            if checkpoint_due
            else "exact_report" if exact_due else "progress_only"
        )
        if boundary.get("payload_profile") != expected_profile:
            raise _fail(f"scheduled boundary {sequence} payload profile mismatch")

        post_bytes = 0
        if exact_due:
            boundary = self._normalize_boundary(
                boundary,
                offset=int(expected["offset"]),
                completed_generation=generation,
                snapshot_sequence=sequence,
                expected_controller_status=expected_clone_status,
            )
            post_bytes = int(
                boundary["tensor_manifest_boundary_bytes"][
                    "post_turnover_checkpoint_projection"
                ]
            )
        else:
            if _strict_int(
                boundary.get("generation_end_reporting_frame_tensor_bytes"),
                field=(
                    f"scheduled boundary {sequence}."
                    "generation_end_reporting_frame_tensor_bytes"
                ),
                minimum=0,
            ) != 0:
                raise _fail(f"scheduled boundary {sequence} has unscheduled report bytes")
            recorded_post_bytes = _strict_int(
                boundary.get("post_turnover_projection_tensor_bytes"),
                field=(
                    f"scheduled boundary {sequence}."
                    "post_turnover_projection_tensor_bytes"
                ),
                minimum=0,
            )
            if checkpoint_due:
                checkpoint_projection = _dict_copy(
                    boundary.get("checkpoint_projection"),
                    field=f"scheduled boundary {sequence}.checkpoint_projection",
                )
                post_bytes = _tensor_manifest_bytes(
                    checkpoint_projection,
                    field=f"scheduled boundary {sequence}.checkpoint_projection",
                )
                if recorded_post_bytes != post_bytes:
                    raise _fail(
                        f"scheduled boundary {sequence} checkpoint byte count mismatch"
                    )
                boundary["checkpoint_projection"] = checkpoint_projection
            elif recorded_post_bytes != 0:
                raise _fail(f"scheduled boundary {sequence} has unscheduled projection bytes")
            boundary["tensor_manifest_boundary_bytes"] = {
                "generation_end_reporting_frame": 0,
                "post_turnover_checkpoint_projection": post_bytes,
                "total": post_bytes,
            }

        if checkpoint_due:
            expected_coordinate = f"generation:{generation + 1}:tick:0"
            if boundary.get("checkpoint_coordinate") != expected_coordinate:
                raise _fail(f"scheduled boundary {sequence} checkpoint coordinate mismatch")
            observer = boundary.get("observer_continuation")
            if not isinstance(observer, Mapping):
                raise _fail(f"scheduled boundary {sequence} has no observer continuation")
        return boundary

    def _normalize_and_validate_scheduled_result(
        self,
        raw: Any,
        *,
        requested: int,
        schedule: Mapping[str, Any],
        expected_terminal_status: str,
        result_retention: str = "all_boundaries",
    ) -> dict[str, Any]:
        result = _dict_copy(raw, field="native scheduled run result")
        if result.get("schema_version") != NATIVE_RESULT_SCHEMA_VERSION:
            raise _fail("native scheduled run result schema mismatch")
        if result.get("terminal_status") != expected_terminal_status:
            raise _fail("native scheduled run terminal status mismatch")
        self._extract_patch_native_instrumentation(result)
        self._validate_closed_phase4_result_fields(
            result,
            terminal_status=expected_terminal_status,
        )
        self._validate_native_identity(result, field="native scheduled run result")
        _require_bool(result, "executed", True)
        _require_bool(result, "native_owner", True)
        _require_bool(result, "hidden_fallback", False)
        _require_bool(result, "runtime_jit", False)
        if self._persistence_mailbox_required:
            if _strict_int(
                result.get("controller_protocol_version"),
                field="controller_protocol_version",
                minimum=1,
            ) != NATIVE_CONTROLLER_PROTOCOL_VERSION:
                raise _fail("scheduled run controller protocol version mismatch")
            _strict_int(
                result.get("boundary_wall_time_ns"),
                field="boundary_wall_time_ns",
                minimum=0,
            )
        _require_bool(
            result,
            "success_marker_allowed",
            expected_terminal_status == "run_complete",
        )
        _require_bool(result, "promotion_result_allowed", False)

        completed = _strict_int(
            result.get("completed_generations"),
            field="completed_generations",
            minimum=0,
        )
        if expected_terminal_status in {
            "awaiting_snapshot_validation",
            "run_complete",
        }:
            if completed != requested:
                raise _fail("scheduled run did not complete every requested generation")
        elif expected_terminal_status == "cancelled":
            if completed < 1 or completed > requested:
                raise _fail("cancelled run is outside a complete-generation boundary")
            cancel_request_id = _strict_ascii_identifier(
                result.get("cancel_request_id"),
                field="cancelled run cancel_request_id",
            )
            cancelled_generation = self._start_generation + completed - 1
            cancelled_sequence = self._initial_boundary_ordinal + completed
            if _strict_int(
                result.get("cancelled_at_generation_boundary"),
                field="cancelled_at_generation_boundary",
                minimum=1,
            ) != self._start_generation + completed:
                raise _fail("cancelled run boundary contradicts completed generations")
            if _strict_int(
                result.get("cancelled_after_completed_generation"),
                field="cancelled_after_completed_generation",
                minimum=0,
            ) != cancelled_generation:
                raise _fail("cancelled run completed-generation marker mismatch")
            if _strict_int(
                result.get("cancelled_snapshot_sequence"),
                field="cancelled_snapshot_sequence",
                minimum=1,
            ) != cancelled_sequence:
                raise _fail("cancelled run snapshot sequence mismatch")
            expected_cancelled_snapshot_id = (
                f"{self._controller_id}:owner:{self._owner_epoch}:"
                f"boundary:{cancelled_sequence}"
            )
            if result.get("cancelled_snapshot_id") != expected_cancelled_snapshot_id:
                raise _fail("cancelled run snapshot identity mismatch")
            if result.get("cancelled_snapshot_completion_disposition") != (
                "cancelled_no_success_or_promotion"
            ):
                raise _fail("cancelled run completion disposition is invalid")
            _require_bool(result, "cancelled_snapshot_durable", True)
            _require_bool(result, "cancellation_checkpoint_requested", True)
            _require_bool(result, "cancellation_checkpoint_persisted", True)
            result["cancel_request_id"] = cancel_request_id
        else:
            raise _fail("scheduled result validator received an unsupported terminal state")

        expected_trajectory_frontier = self._start_generation + completed - 1
        expected_scalars = {
            "start_generation": self._start_generation,
            "requested_generations": requested,
            "trajectory_commit_frontier": expected_trajectory_frontier,
            "next_generation": self._start_generation + completed,
            "next_tick": 0,
            "controller_invocations": 1,
            "python_generation_dispatches": 0,
            "native_generation_transactions": completed,
            "turnover_commits": completed,
            "final_snapshot_sequence": self._initial_boundary_ordinal + completed,
        }
        expectations, projection_exports = self._scheduled_boundary_expectations(
            completed=completed,
            requested=requested,
            schedule=schedule,
            terminal_cancelled=expected_terminal_status == "cancelled",
        )
        expected_scalars["boundary_snapshot_count"] = len(expectations)
        expected_scalars["projection_export_count"] = projection_exports
        for field, expected_value in expected_scalars.items():
            if _strict_int(result.get(field), field=field, minimum=-1) != expected_value:
                raise _fail(
                    f"scheduled run {field} mismatch: "
                    f"expected={expected_value} actual={result.get(field)}"
                )
        if _strict_int(
            result.get("observer_live_owner_exports"),
            field="observer_live_owner_exports",
            minimum=0,
        ) != 0:
            raise _fail("scheduled observers read the live resident owner")

        acknowledgement_expectations = [
            row for row in expectations if row["persistence_ack_required"]
        ]
        expected_snapshot_frontier = (
            int(acknowledgement_expectations[-1]["generation"])
            if acknowledgement_expectations
            else self._start_generation - 1
        )
        if _strict_int(
            result.get("snapshot_frontier"),
            field="snapshot_frontier",
            minimum=-1,
        ) != expected_snapshot_frontier:
            raise _fail("scheduled run snapshot frontier mismatch")
        if _strict_int(
            result.get("persistence_acknowledgement_count"),
            field="persistence_acknowledgement_count",
            minimum=0,
        ) != len(acknowledgement_expectations):
            raise _fail("scheduled run persistence acknowledgement count mismatch")

        retained_expectations = (
            expectations if result_retention == "all_boundaries" else expectations[-1:]
        )
        raw_boundaries = result.get("boundaries")
        if not isinstance(raw_boundaries, list) or len(raw_boundaries) != len(
            retained_expectations
        ):
            raise _fail("scheduled run boundary count mismatch")
        normalized_boundaries = [
            self._normalize_scheduled_boundary(
                raw_boundary,
                expected=expected,
                schedule=schedule,
            )
            for raw_boundary, expected in zip(
                raw_boundaries, retained_expectations, strict=True
            )
        ]
        total_boundary_bytes = sum(
            int(boundary["tensor_manifest_boundary_bytes"]["total"])
            for boundary in normalized_boundaries
        )
        native_projection_bytes = _strict_int(
            result.get("projection_tensor_bytes"),
            field="projection_tensor_bytes",
            minimum=0,
        )
        if (
            result_retention == "all_boundaries"
            and native_projection_bytes != total_boundary_bytes
        ) or (
            result_retention == "terminal_only"
            and native_projection_bytes < total_boundary_bytes
        ):
            raise _fail("scheduled run projection tensor byte count mismatch")

        if expected_terminal_status == "cancelled":
            if not normalized_boundaries:
                raise _fail("cancelled run has no authoritative boundary")
            self._validate_cancelled_terminal_receipts(
                result,
                cancelled_boundary=normalized_boundaries[-1],
                completed=completed,
            )

        result["boundaries"] = normalized_boundaries
        result["result_retention"] = result_retention
        result["retained_boundary_count"] = len(normalized_boundaries)
        result["retained_projection_tensor_bytes"] = total_boundary_bytes
        result["trajectory_dispatch_count"] = 1
        result["boundary_tensor_manifest_bytes"] = total_boundary_bytes
        self._boundary_tensor_manifest_bytes = total_boundary_bytes
        if expected_terminal_status in {
            "awaiting_snapshot_validation",
            "run_complete",
        }:
            final_boundary = normalized_boundaries[-1]
            final_expectation = retained_expectations[-1]
            if final_expectation["exact_due"]:
                final_projection = final_boundary[
                    "post_turnover_checkpoint_projection"
                ]
            elif (
                self._compact_reporting_only
                and final_expectation["checkpoint_due"]
            ):
                final_projection = final_boundary["checkpoint_projection"]
            else:
                raise _fail(
                    "scheduled completed run has no exact final projection"
                )
            result["final_checkpoint_projection"] = final_projection
            result["final_checkpoint_projection_digest_manifests"] = (
                _projection_digest_manifests(final_projection)
            )
        return result

    def _normalize_and_validate_result(
        self,
        raw: Any,
        *,
        requested: int,
        expected_terminal_status: str,
    ) -> dict[str, Any]:
        result = _dict_copy(raw, field="native run result")
        if result.get("schema_version") != NATIVE_RESULT_SCHEMA_VERSION:
            raise _fail("native run result schema mismatch")
        terminal_status = result.get("terminal_status")
        if terminal_status != expected_terminal_status:
            raise _fail(
                "native run terminal status mismatch: "
                f"expected={expected_terminal_status!r} actual={terminal_status!r}"
            )
        self._extract_patch_native_instrumentation(result)
        self._validate_closed_phase4_result_fields(
            result,
            terminal_status=expected_terminal_status,
        )
        self._validate_native_identity(result, field="native run result")
        _require_bool(result, "executed", True)
        _require_bool(result, "native_owner", True)
        _require_bool(result, "hidden_fallback", False)
        _require_bool(result, "runtime_jit", False)
        if self._persistence_mailbox_required:
            if _strict_int(
                result.get("controller_protocol_version"),
                field="controller_protocol_version",
                minimum=1,
            ) != NATIVE_CONTROLLER_PROTOCOL_VERSION:
                raise _fail("native run controller protocol version mismatch")
            _strict_int(
                result.get("boundary_wall_time_ns"),
                field="boundary_wall_time_ns",
                minimum=0,
            )

        completed = _strict_int(
            result.get("completed_generations"),
            field="completed_generations",
            minimum=0,
        )
        if completed != requested:
            raise _fail(
                "completed generation count mismatch: "
                f"requested={requested} completed={completed}"
            )
        expected_scalars = {
            "start_generation": self._start_generation,
            "requested_generations": requested,
            "trajectory_commit_frontier": self._start_generation + requested - 1,
            "next_generation": self._start_generation + requested,
            "next_tick": 0,
            "controller_invocations": 1,
            "python_generation_dispatches": 0,
            "native_generation_transactions": requested,
            "turnover_commits": requested,
            "boundary_snapshot_count": requested,
            "projection_export_count": requested * 2,
            "final_snapshot_sequence": requested,
        }
        for field, expected in expected_scalars.items():
            actual = _strict_int(result.get(field), field=field, minimum=0)
            if actual != expected:
                raise _fail(f"{field} mismatch: expected={expected} actual={actual}")
        expected_snapshot_frontier = (
            self._start_generation + requested - 1
            if expected_terminal_status == "run_complete"
            else self._start_generation - 1
        )
        if _strict_int(
            result.get("snapshot_frontier"),
            field="snapshot_frontier",
            minimum=-1,
        ) != expected_snapshot_frontier:
            raise _fail("snapshot_frontier does not match validation state")
        _require_bool(
            result,
            "success_marker_allowed",
            expected_terminal_status == "run_complete",
        )

        boundaries = result.get("boundaries")
        if not isinstance(boundaries, list) or len(boundaries) != requested:
            raise _fail("native boundary count does not match requested generations")
        normalized_boundaries = []
        total_boundary_bytes = 0
        for offset, raw_boundary in enumerate(boundaries):
            boundary = self._normalize_boundary(raw_boundary, offset=offset)
            byte_manifest = boundary["tensor_manifest_boundary_bytes"]
            total_boundary_bytes += int(byte_manifest["total"])
            normalized_boundaries.append(boundary)

        native_total = _strict_int(
            result.get("projection_tensor_bytes"),
            field="projection_tensor_bytes",
            minimum=0,
        )
        if native_total != total_boundary_bytes:
            raise _fail(
                "projection tensor byte count mismatch: "
                f"native={native_total} exact={total_boundary_bytes}"
            )

        final_projection = normalized_boundaries[-1][
            "post_turnover_checkpoint_projection"
        ]
        final_manifests = _projection_digest_manifests(final_projection)
        last_post_manifests = normalized_boundaries[-1][
            "checkpoint_projection_digest_manifests"
        ]
        if final_manifests != last_post_manifests:
            raise _fail("final checkpoint projection does not match the last boundary")

        self._boundary_tensor_manifest_bytes = total_boundary_bytes
        result["boundaries"] = normalized_boundaries
        # This is an in-result alias to the already detached last boundary, not
        # another owner read or tensor copy.
        result["final_checkpoint_projection"] = final_projection
        result["final_checkpoint_projection_digest_manifests"] = final_manifests
        result["trajectory_dispatch_count"] = 1
        result["boundary_tensor_manifest_bytes"] = total_boundary_bytes
        return result

    def _normalize_snapshot_validation_receipt(
        self,
        raw: Any,
        *,
        provisional: Mapping[str, Any],
        prepared_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt = _closed_dict(
            _dict_copy(raw, field="snapshot validation receipt"),
            field="snapshot validation receipt",
            required_fields=(
                _SNAPSHOT_VALIDATION_SETTLEMENT_RECEIPT_FIELDS
                if prepared_receipt is not None
                else _SNAPSHOT_VALIDATION_RECEIPT_FIELDS
            ),
        )
        if (
            receipt.get("schema_version")
            != NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION
        ):
            raise _fail("snapshot validation receipt schema mismatch")
        self._validate_native_identity(
            receipt,
            field="snapshot validation receipt",
        )
        _require_bool(receipt, "acknowledged", True)
        if receipt.get("disposition") != "accepted":
            raise _fail("snapshot validation receipt was not accepted")
        if receipt.get("terminal_status") != "run_complete":
            raise _fail("snapshot validation receipt is not terminal run_complete")
        _require_bool(receipt, "success_marker_allowed", True)

        provisional_manifests = _dict_copy(
            provisional.get("final_checkpoint_projection_digest_manifests"),
            field="provisional final checkpoint projection digest manifests",
        )
        provisional_source = _dict_copy(
            provisional_manifests.get("source_projection"),
            field="provisional final source-projection digest manifest",
        )
        expected_source_projection_digest = _strict_lowercase_sha256(
            provisional_source.get("digest"),
            field="provisional final source-projection digest",
        )
        receipt_source_projection_digest = _strict_lowercase_sha256(
            receipt.get("validated_source_projection_digest"),
            field="snapshot validation receipt.validated_source_projection_digest",
        )
        if receipt_source_projection_digest != expected_source_projection_digest:
            raise _fail(
                "snapshot validation receipt source-projection digest mismatch"
            )

        trajectory_frontier = _strict_int(
            provisional.get("trajectory_commit_frontier"),
            field="provisional trajectory_commit_frontier",
            minimum=-1,
        )
        receipt_trajectory_frontier = _strict_int(
            receipt.get("trajectory_commit_frontier"),
            field="snapshot validation receipt.trajectory_commit_frontier",
            minimum=-1,
        )
        snapshot_frontier = _strict_int(
            receipt.get("snapshot_frontier"),
            field="snapshot validation receipt.snapshot_frontier",
            minimum=-1,
        )
        if (
            receipt_trajectory_frontier != trajectory_frontier
            or snapshot_frontier != trajectory_frontier
        ):
            raise _fail("snapshot validation receipt frontier mismatch")

        final_snapshot_sequence = _strict_int(
            provisional.get("final_snapshot_sequence"),
            field="provisional final_snapshot_sequence",
            minimum=1,
        )
        if _strict_int(
            receipt.get("final_snapshot_sequence"),
            field="snapshot validation receipt.final_snapshot_sequence",
            minimum=1,
        ) != final_snapshot_sequence:
            raise _fail("snapshot validation receipt sequence mismatch")
        if prepared_receipt is not None:
            for field in ("settlement_token", "prepared_receipt_digest_sha256"):
                value = _strict_lowercase_sha256(
                    receipt.get(field),
                    field=f"snapshot validation receipt.{field}",
                )
                if value != prepared_receipt.get(field):
                    raise _fail(
                        f"snapshot validation receipt {field} contradicts prepare"
                    )
        return receipt

    def _normalize_snapshot_validation_prepare_receipt(
        self,
        raw: Any,
        *,
        provisional: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = _closed_dict(
            _dict_copy(raw, field="snapshot validation prepare receipt"),
            field="snapshot validation prepare receipt",
            required_fields=_SNAPSHOT_VALIDATION_PREPARE_RECEIPT_FIELDS,
        )
        if receipt.get("schema_version") != (
            NATIVE_SNAPSHOT_VALIDATION_PREPARE_RECEIPT_SCHEMA_VERSION
        ):
            raise _fail("snapshot validation prepare receipt schema mismatch")
        self._validate_native_identity(
            receipt,
            field="snapshot validation prepare receipt",
        )
        _require_bool(receipt, "prepared", True)
        if receipt.get("disposition") != "accepted":
            raise _fail("snapshot validation prepare receipt was not accepted")
        if receipt.get("terminal_status") != (
            "awaiting_snapshot_validation_settlement"
        ):
            raise _fail("snapshot validation prepare receipt is already terminal")
        _require_bool(receipt, "success_marker_allowed", False)

        manifests = _dict_copy(
            provisional.get("final_checkpoint_projection_digest_manifests"),
            field="provisional final checkpoint projection digest manifests",
        )
        source_manifest = _dict_copy(
            manifests.get("source_projection"),
            field="provisional final source-projection digest manifest",
        )
        expected_source_digest = _strict_lowercase_sha256(
            source_manifest.get("digest"),
            field="provisional final source-projection digest",
        )
        if _strict_lowercase_sha256(
            receipt.get("validated_source_projection_digest"),
            field="snapshot validation prepare receipt source-projection digest",
        ) != expected_source_digest:
            raise _fail(
                "snapshot validation prepare receipt source-projection digest mismatch"
            )

        expected_trajectory_frontier = _strict_int(
            provisional.get("trajectory_commit_frontier"),
            field="provisional trajectory_commit_frontier",
            minimum=-1,
        )
        expected_snapshot_frontier = (
            _strict_int(
                provisional.get("snapshot_frontier"),
                field="provisional snapshot_frontier",
                minimum=-1,
            )
            if self._persistence_mailbox_required
            else expected_trajectory_frontier
        )
        if _strict_int(
            receipt.get("trajectory_commit_frontier"),
            field="snapshot validation prepare trajectory_commit_frontier",
            minimum=-1,
        ) != expected_trajectory_frontier:
            raise _fail("snapshot validation prepare trajectory frontier mismatch")
        if _strict_int(
            receipt.get("proposed_snapshot_frontier"),
            field="snapshot validation prepare proposed_snapshot_frontier",
            minimum=-1,
        ) != expected_snapshot_frontier:
            raise _fail("snapshot validation prepare snapshot frontier mismatch")
        expected_sequence = _strict_int(
            provisional.get("final_snapshot_sequence"),
            field="provisional final_snapshot_sequence",
            minimum=1,
        )
        if _strict_int(
            receipt.get("final_snapshot_sequence"),
            field="snapshot validation prepare final_snapshot_sequence",
            minimum=1,
        ) != expected_sequence:
            raise _fail("snapshot validation prepare sequence mismatch")
        _strict_lowercase_sha256(
            receipt.get("settlement_token"),
            field="snapshot validation settlement token",
        )
        _strict_lowercase_sha256(
            receipt.get("prepared_receipt_digest_sha256"),
            field="snapshot validation prepared receipt digest",
        )
        return receipt

    def _settle_snapshot_validation_two_phase(
        self,
        *,
        provisional: Mapping[str, Any],
        source_projection_digest: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prepare = getattr(self._native_controller, "prepare_snapshot_validation", None)
        commit = getattr(self._native_controller, "commit_snapshot_validation", None)
        if not callable(prepare) or not callable(commit):
            raise _fail("native controller lacks two-phase snapshot settlement")
        prepared = self._normalize_snapshot_validation_prepare_receipt(
            prepare(source_projection_digest),
            provisional=provisional,
        )
        self._validate_native_manifest(
            expected_status="awaiting_snapshot_validation_settlement"
        )
        committed = commit(
            {
                "api_version": 1,
                "expected_controller_id": self._controller_id,
                "expected_owner_epoch": self._owner_epoch,
                "settlement_token": prepared["settlement_token"],
                "prepared_receipt_digest_sha256": prepared[
                    "prepared_receipt_digest_sha256"
                ],
            }
        )
        self._validate_native_manifest(expected_status="run_complete")
        receipt = self._normalize_snapshot_validation_receipt(
            committed,
            provisional=provisional,
            prepared_receipt=prepared,
        )
        return prepared, receipt

    def _normalize_failed_result(self, raw: Any) -> dict[str, Any]:
        result = _dict_copy(raw, field="native failed run result")
        if result.get("schema_version") != NATIVE_RESULT_SCHEMA_VERSION:
            raise _fail("native failed run result schema mismatch")
        if result.get("terminal_status") != "failed":
            raise _fail("native failed run result is not terminal failed")
        self._extract_patch_native_instrumentation(result)
        self._validate_closed_phase4_result_fields(
            result,
            terminal_status="failed",
        )
        self._validate_native_identity(result, field="native failed run result")
        _require_bool(result, "executed", False)
        _require_bool(result, "native_owner", True)
        _require_bool(result, "hidden_fallback", False)
        _require_bool(result, "runtime_jit", False)
        _require_bool(result, "success_marker_allowed", False)
        root_failure = _dict_copy(
            result.get("root_failure"),
            field="native failed run root_failure",
        )
        if self._persistence_mailbox_required:
            _closed_dict(
                root_failure,
                field="native Phase 4 root_failure",
                required_fields=_PHASE4_ROOT_FAILURE_FIELDS,
            )
        if _strict_int(
            root_failure.get("error_sequence"),
            field="root_failure.error_sequence",
            minimum=1,
        ) != 1:
            raise _fail("native failed run root failure is not first-error-wins")
        if not str(root_failure.get("error_code", "") or ""):
            raise _fail("native failed run root failure has no error code")
        if not str(root_failure.get("message", "") or ""):
            raise _fail("native failed run root failure has no message")
        self._validate_native_identity(
            root_failure,
            field="native failed run root failure",
        )
        trajectory_frontier = _strict_int(
            result.get("trajectory_commit_frontier"),
            field="native failed run trajectory_commit_frontier",
            minimum=-1,
        )
        snapshot_frontier = _strict_int(
            result.get("snapshot_frontier"),
            field="native failed run snapshot_frontier",
            minimum=-1,
        )
        if snapshot_frontier > trajectory_frontier:
            raise _fail("native failed run snapshot frontier exceeds trajectory frontier")
        boundaries = result.get("boundaries")
        if not isinstance(boundaries, list):
            raise _fail("native failed run boundaries are not a list")
        for offset, boundary in enumerate(boundaries):
            if not isinstance(boundary, Mapping):
                raise _fail(f"native failed run boundaries[{offset}] is not a mapping")
            self._validate_native_identity(
                boundary,
                field=f"native failed run boundaries[{offset}]",
            )
        result["root_failure"] = root_failure
        result["boundaries"] = _detached_copy(boundaries)
        return result

    def _normalize_policy_mutation_tape_boundary_authority(
        self,
        boundary: Mapping[str, Any],
        post_turnover_projection: Mapping[str, Any],
        *,
        offset: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        path = f"boundaries[{offset}].policy_mutation_tape_authority"
        boundary_tape = boundary.get("policy_mutation_tape")
        resident_surface_present = isinstance(boundary_tape, Mapping) and bool(
            frozenset(boundary_tape)
            & (
                frozenset(_POLICY_MUTATION_TAPE_FIELDS)
                | {
                    _POLICY_MUTATION_TAPE_CURSOR_FIELD,
                    _POLICY_MUTATION_TAPE_PACKED_POST_FIELD,
                }
            )
        )
        native_projection_present = post_turnover_projection.get(
            "schema_version"
        ) in NATIVE_CHECKPOINT_PROJECTION_SCHEMA_VERSIONS
        if "policy_mutation_tape_authority" not in boundary:
            if not resident_surface_present and not native_projection_present:
                return None
            raise _fail(f"{path} required authority is absent")
        authority = _closed_dict(
            _dict_copy(boundary["policy_mutation_tape_authority"], field=path),
            field=path,
            required_fields=_POLICY_MUTATION_TAPE_AUTHORITY_FIELDS,
        )
        if authority["schema_version"] != (
            NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION
        ):
            raise _fail(f"{path}.schema_version mismatch")
        if authority["source_version"] != POLICY_MUTATION_TAPE_SCHEMA_VERSION:
            raise _fail(f"{path}.source_version mismatch")
        _require_bool(authority, "authority_complete", True)
        _require_bool(authority, "export_non_consuming", True)
        cursor = _strict_int(
            authority["cursor"], field=f"{path}.cursor", minimum=0
        )

        required_tape_fields = frozenset(_POLICY_MUTATION_TAPE_FIELDS)
        consumed_shapes = _closed_dict(
            authority["consumed_shapes"],
            field=f"{path}.consumed_shapes",
            required_fields=required_tape_fields,
        )
        tensor_sha256 = _closed_dict(
            authority["tensor_sha256"],
            field=f"{path}.tensor_sha256",
            required_fields=required_tape_fields,
        )
        if not isinstance(boundary_tape, Mapping):
            raise _fail(f"{path} requires boundary policy mutation tape tensors")

        tape: dict[str, Any] = {}
        capacity: int | None = None
        resident_shapes: dict[str, tuple[int, ...]] = {}
        common_fields = {
            name: spec
            for name, spec in _POLICY_MUTATION_TAPE_FIELDS.items()
            if name != _POLICY_MUTATION_TAPE_DENSE_POST_FIELD
        }
        for name, (expected_dtype, expected_rank) in common_fields.items():
            tensor = boundary_tape.get(name)
            field_path = f"{path}.{name}"
            if not hasattr(tensor, "detach") or not hasattr(tensor, "shape"):
                raise _fail(f"{field_path} resident tensor is absent")
            if str(tensor.dtype) != expected_dtype:
                raise _fail(
                    f"{field_path} dtype mismatch: "
                    f"expected={expected_dtype} actual={tensor.dtype}"
                )
            shape = tuple(int(size) for size in tensor.shape)
            if len(shape) != expected_rank:
                raise _fail(
                    f"{field_path} rank mismatch: "
                    f"expected={expected_rank} actual={len(shape)}"
                )
            if capacity is None:
                capacity = shape[0]
            elif shape[0] != capacity:
                raise _fail(f"{path} resident field capacities disagree")
            if cursor > shape[0]:
                raise _fail(
                    f"{path}.cursor exceeds resident capacity for {name}: "
                    f"cursor={cursor} capacity={shape[0]}"
                )
            resident_shapes[name] = shape
            prefix = tensor.detach().narrow(0, 0, cursor).contiguous()
            actual_shape = [int(size) for size in prefix.shape]
            recorded_shape = consumed_shapes[name]
            if type(recorded_shape) is not list or any(
                type(size) is not int or size < 0 for size in recorded_shape
            ):
                raise _fail(f"{path}.consumed_shapes.{name} is malformed")
            if recorded_shape != actual_shape:
                raise _fail(
                    f"{path}.consumed_shapes.{name} contradicts the consumed prefix"
                )
            recorded_digest = _strict_lowercase_sha256(
                tensor_sha256[name], field=f"{path}.tensor_sha256.{name}"
            )
            actual_digest = hashlib.sha256(
                prefix.to("cpu").numpy().tobytes(order="C")
            ).hexdigest()
            if recorded_digest != actual_digest:
                raise _fail(
                    f"{path}.tensor_sha256.{name} contradicts the consumed prefix"
                )
            tape[name] = prefix

        has_dense_post = _POLICY_MUTATION_TAPE_DENSE_POST_FIELD in boundary_tape
        has_packed_post = _POLICY_MUTATION_TAPE_PACKED_POST_FIELD in boundary_tape
        if has_dense_post == has_packed_post:
            raise _fail(
                f"{path} must contain exactly one resident post-RNG state tensor"
            )
        resident_post_name = (
            _POLICY_MUTATION_TAPE_PACKED_POST_FIELD
            if has_packed_post
            else _POLICY_MUTATION_TAPE_DENSE_POST_FIELD
        )
        expected_dtype, expected_rank = (
            _POLICY_MUTATION_TAPE_PACKED_POST_SPEC
            if has_packed_post
            else _POLICY_MUTATION_TAPE_FIELDS[_POLICY_MUTATION_TAPE_DENSE_POST_FIELD]
        )
        post_tensor = boundary_tape.get(resident_post_name)
        post_path = f"{path}.{resident_post_name}"
        if not hasattr(post_tensor, "detach") or not hasattr(post_tensor, "shape"):
            raise _fail(f"{post_path} resident tensor is absent")
        if str(post_tensor.dtype) != expected_dtype:
            raise _fail(
                f"{post_path} dtype mismatch: "
                f"expected={expected_dtype} actual={post_tensor.dtype}"
            )
        post_shape = tuple(int(size) for size in post_tensor.shape)
        if len(post_shape) != expected_rank:
            raise _fail(
                f"{post_path} rank mismatch: "
                f"expected={expected_rank} actual={len(post_shape)}"
            )
        replay_backed_post_state = has_packed_post and post_shape == (0, 0)
        if (
            has_packed_post
            and not replay_backed_post_state
            and post_shape[1] != _TORCH_CPU_RNG_PACKED_STATE_BYTES_V1
        ):
            raise _fail(f"{post_path} must be uint8[N,2560]")
        if replay_backed_post_state and capacity is None:
            raise _fail(f"{post_path} replay route requires resident tape fields")
        if capacity is None:
            capacity = post_shape[0]
        elif not replay_backed_post_state and post_shape[0] != capacity:
            raise _fail(f"{path} resident field capacities disagree")
        if not replay_backed_post_state and cursor > post_shape[0]:
            raise _fail(
                f"{path}.cursor exceeds resident capacity for {resident_post_name}: "
                f"cursor={cursor} capacity={post_shape[0]}"
            )
        post_prefix = (
            post_tensor.detach().contiguous()
            if replay_backed_post_state
            else post_tensor.detach().narrow(0, 0, cursor).contiguous()
        )
        recorded_post_shape = consumed_shapes[_POLICY_MUTATION_TAPE_DENSE_POST_FIELD]
        if type(recorded_post_shape) is not list or any(
            type(size) is not int or size < 0 for size in recorded_post_shape
        ):
            raise _fail(
                f"{path}.consumed_shapes."
                f"{_POLICY_MUTATION_TAPE_DENSE_POST_FIELD} is malformed"
            )
        expected_post_shape = (
            [cursor, _TORCH_CPU_RNG_FULL_STATE_BYTES_V1]
            if has_packed_post
            else [int(size) for size in post_prefix.shape]
        )
        if recorded_post_shape != expected_post_shape:
            raise _fail(
                f"{path}.consumed_shapes."
                f"{_POLICY_MUTATION_TAPE_DENSE_POST_FIELD} contradicts the "
                "canonical consumed prefix"
            )
        _strict_lowercase_sha256(
            tensor_sha256[_POLICY_MUTATION_TAPE_DENSE_POST_FIELD],
            field=f"{path}.tensor_sha256.{_POLICY_MUTATION_TAPE_DENSE_POST_FIELD}",
        )
        if not has_packed_post:
            actual_digest = hashlib.sha256(
                post_prefix.to("cpu").numpy().tobytes(order="C")
            ).hexdigest()
            if (
                tensor_sha256[_POLICY_MUTATION_TAPE_DENSE_POST_FIELD]
                != actual_digest
            ):
                raise _fail(
                    f"{path}.tensor_sha256."
                    f"{_POLICY_MUTATION_TAPE_DENSE_POST_FIELD} contradicts the "
                    "consumed prefix"
                )
        tape[resident_post_name] = post_prefix

        delta_shape = resident_shapes["policy_birth_delta_tape"]
        mask_shape = resident_shapes["policy_birth_mutation_mask_tape"]
        crossover_shape = resident_shapes["policy_birth_crossover_mask_tape"]
        if mask_shape != delta_shape:
            raise _fail(
                f"{path}.policy_birth_mutation_mask_tape shape must match "
                "the delta tape"
            )
        if crossover_shape[0] != delta_shape[0]:
            raise _fail(
                f"{path}.policy_birth_crossover_mask_tape capacity must match "
                "the delta tape"
            )
        if delta_shape[1] not in (1, crossover_shape[1]):
            raise _fail(
                f"{path}.policy_birth_delta_tape width must be compact width "
                "1 or match the crossover tape"
            )

        cursor_tensor = boundary_tape.get(_POLICY_MUTATION_TAPE_CURSOR_FIELD)
        cursor_path = f"{path}.{_POLICY_MUTATION_TAPE_CURSOR_FIELD}"
        if not hasattr(cursor_tensor, "detach") or not hasattr(
            cursor_tensor, "shape"
        ):
            raise _fail(f"{cursor_path} resident tensor is absent")
        if str(cursor_tensor.dtype) != "torch.int64" or tuple(
            int(size) for size in cursor_tensor.shape
        ) != (1,):
            raise _fail(f"{cursor_path} must be int64[1]")
        if int(cursor_tensor.detach().to("cpu").item()) != cursor:
            raise _fail(f"{cursor_path} contradicts authority cursor")
        tape[_POLICY_MUTATION_TAPE_CURSOR_FIELD] = cursor_tensor.detach()

        try:
            descriptor_digest = policy_mutation_tape_digest_from_tensor_sha256(
                cursor, consumed_shapes, tensor_sha256
            )
            resident_digest = (
                descriptor_digest
                if replay_backed_post_state
                else policy_mutation_tape_digest_sha256(tape)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _fail(f"{path} violates the consumed-prefix contract: {exc}") from exc
        if resident_digest != descriptor_digest:
            raise _fail(f"{path} canonical consumed-prefix digest mismatch")
        return authority, tape

    def _normalize_native_compact_digest_authority(
        self,
        boundary: Mapping[str, Any],
        post_turnover_projection: Mapping[str, Any],
        policy_mutation_tape: Mapping[str, Any] | None,
        *,
        offset: int,
    ) -> dict[str, Any] | None:
        field = "native_compact_digest_authority"
        path = f"boundaries[{offset}].{field}"
        if field not in boundary:
            return None
        authority = _closed_dict(
            _dict_copy(boundary[field], field=path),
            field=path,
            required_fields=_NATIVE_COMPACT_DIGEST_AUTHORITY_FIELDS,
        )
        if authority["schema_version"] != (
            NATIVE_COMPACT_DIGEST_AUTHORITY_SCHEMA_VERSION
        ):
            raise _fail(f"{path}.schema_version mismatch")
        _require_bool(
            authority, "authority_complete_for_declared_fields", True
        )
        _require_bool(authority, "python_digest_execution", False)
        _require_bool(authority, "full_checkpoint_projection_used", False)
        if _strict_int(
            authority["declared_digest_count"],
            field=f"{path}.declared_digest_count",
            minimum=0,
        ) != 3:
            raise _fail(f"{path}.declared_digest_count must equal 3")
        if policy_mutation_tape is None:
            raise _fail(f"{path} requires the validated policy mutation tape")

        tensors = post_turnover_projection.get("tensors")
        if not isinstance(tensors, Mapping):
            raise _fail(f"{path} requires post-turnover tensors")
        active_ids = tensors.get("agent_ids_by_alive_order")
        if not hasattr(active_ids, "numel"):
            raise _fail(f"{path} requires resident active agent IDs")
        active_count = int(active_ids.numel())
        event_cursor = _strict_int(
            post_turnover_projection.get("event_cursor"),
            field=f"{path}.post_turnover_event_cursor",
            minimum=0,
        )
        tape_cursor_tensor = policy_mutation_tape.get(
            _POLICY_MUTATION_TAPE_CURSOR_FIELD
        )
        if not hasattr(tape_cursor_tensor, "item"):
            raise _fail(f"{path} requires the resident policy tape cursor")
        tape_cursor = int(tape_cursor_tensor.detach().to("cpu").item())
        for name, expected in (
            ("lifecycle_event_cursor", event_cursor),
            ("active_agent_count", active_count),
            ("policy_mutation_tape_cursor", tape_cursor),
        ):
            actual = _strict_int(
                authority[name], field=f"{path}.{name}", minimum=0
            )
            if actual != expected:
                raise _fail(
                    f"{path}.{name} contradicts resident authority: "
                    f"expected={expected} actual={actual}"
                )

        try:
            expected_digests = {
                "events_active_prefix_digest_sha256": (
                    typed_lifecycle_event_digest(
                        typed_lifecycle_events_from_projection(
                            post_turnover_projection
                        )
                    )
                ),
                "identity_active_agent_ids_digest_sha256": (
                    active_agent_ids_digest_sha256(active_ids)
                ),
                "rng_policy_mutation_tape_digest_sha256": (
                    policy_mutation_tape_digest_sha256(policy_mutation_tape)
                ),
            }
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise _fail(f"{path} cannot validate resident authority: {exc}") from exc
        for name, expected in expected_digests.items():
            actual = _strict_lowercase_sha256(
                authority[name], field=f"{path}.{name}"
            )
            if actual != expected:
                raise _fail(f"{path}.{name} contradicts resident authority")
        return authority

    def _normalize_boundary(
        self,
        raw: Any,
        *,
        offset: int,
        completed_generation: int | None = None,
        snapshot_sequence: int | None = None,
        expected_controller_status: str = (
            "generation_complete_ready_to_resolve_terminal"
        ),
    ) -> dict[str, Any]:
        boundary = _dict_copy(raw, field=f"boundaries[{offset}]")
        if boundary.get("schema_version") != NATIVE_BOUNDARY_SCHEMA_VERSION:
            raise _fail(f"boundaries[{offset}] schema mismatch")
        self._validate_native_identity(boundary, field=f"boundaries[{offset}]")
        resolved_completed_generation = (
            self._start_generation + offset
            if completed_generation is None
            else completed_generation
        )
        resolved_snapshot_sequence = (
            offset + 1 if snapshot_sequence is None else snapshot_sequence
        )
        expected_scalars = {
            "snapshot_sequence": resolved_snapshot_sequence,
            "completed_generation": resolved_completed_generation,
            "next_generation": resolved_completed_generation + 1,
            "tick": 0,
        }
        for field, expected in expected_scalars.items():
            actual = _strict_int(
                boundary.get(field),
                field=f"boundaries[{offset}].{field}",
                minimum=0,
            )
            if actual != expected:
                raise _fail(
                    f"boundaries[{offset}].{field} mismatch: "
                    f"expected={expected} actual={actual}"
                )
        if boundary.get("phase") != "generation_complete":
            raise _fail(f"boundaries[{offset}] has an invalid phase")
        if boundary.get("controller_status_at_clone") != expected_controller_status:
            raise _fail(f"boundaries[{offset}] has an invalid clone status")
        expected_snapshot_id = (
            f"{self._controller_id}:owner:{self._owner_epoch}:"
            f"boundary:{resolved_snapshot_sequence}"
        )
        if boundary.get("snapshot_id") != expected_snapshot_id:
            raise _fail(f"boundaries[{offset}] snapshot identity mismatch")
        if boundary.get("generation_end_reporting_frame_role") != (
            "transaction_local_generation_end_values_folded_after_turnover_commit"
        ):
            raise _fail(f"boundaries[{offset}] reporting-frame role mismatch")

        generation_result = _dict_copy(
            boundary.get("generation_result"),
            field=f"boundaries[{offset}].generation_result",
        )
        turnover_result = _dict_copy(
            boundary.get("turnover_result"),
            field=f"boundaries[{offset}].turnover_result",
        )
        _require_bool(generation_result, "executed", True)
        _require_bool(generation_result, "hidden_fallback", False)
        _require_bool(turnover_result, "executed", True)
        _require_bool(turnover_result, "hidden_fallback", False)

        pre = _dict_copy(
            boundary.get("generation_end_reporting_frame"),
            field=f"boundaries[{offset}].generation_end_reporting_frame",
        )
        post = _dict_copy(
            boundary.get("post_turnover_checkpoint_projection"),
            field=f"boundaries[{offset}].post_turnover_checkpoint_projection",
        )
        pre_bytes = _tensor_manifest_bytes(
            pre,
            field=f"boundaries[{offset}].generation_end_reporting_frame",
        )
        post_bytes = _tensor_manifest_bytes(
            post,
            field=f"boundaries[{offset}].post_turnover_checkpoint_projection",
        )
        recorded_pre = _strict_int(
            boundary.get("generation_end_reporting_frame_tensor_bytes"),
            field=f"boundaries[{offset}].generation_end_reporting_frame_tensor_bytes",
            minimum=0,
        )
        recorded_post = _strict_int(
            boundary.get("post_turnover_projection_tensor_bytes"),
            field=f"boundaries[{offset}].post_turnover_projection_tensor_bytes",
            minimum=0,
        )
        if recorded_pre != pre_bytes or recorded_post != post_bytes:
            raise _fail(f"boundaries[{offset}] native tensor byte counters are not exact")

        normalized_policy_tape = (
            self._normalize_policy_mutation_tape_boundary_authority(
                boundary, post, offset=offset
            )
        )
        normalized_native_compact_digests = (
            self._normalize_native_compact_digest_authority(
                boundary,
                post,
                (
                    normalized_policy_tape[1]
                    if normalized_policy_tape is not None
                    else None
                ),
                offset=offset,
            )
        )

        boundary["generation_result"] = generation_result
        boundary["turnover_result"] = turnover_result
        boundary["generation_end_reporting_frame"] = pre
        boundary["post_turnover_checkpoint_projection"] = post
        if normalized_policy_tape is not None:
            policy_authority, policy_tape = normalized_policy_tape
            boundary["policy_mutation_tape_authority"] = policy_authority
            boundary["policy_mutation_tape"] = policy_tape
        if normalized_native_compact_digests is not None:
            boundary["native_compact_digest_authority"] = (
                normalized_native_compact_digests
            )
        boundary["reporting_projection_digest_manifests"] = _projection_digest_manifests(pre)
        boundary["checkpoint_projection_digest_manifests"] = _projection_digest_manifests(post)
        boundary["tensor_manifest_boundary_bytes"] = {
            "generation_end_reporting_frame": pre_bytes,
            "post_turnover_checkpoint_projection": post_bytes,
            "total": pre_bytes + post_bytes,
        }
        boundary["snapshot_digest_pending_adapter"] = False
        return boundary


__all__ = [
    "NativeRunController",
    "NativeRunControllerError",
    "SCHEMA_VERSION",
]
