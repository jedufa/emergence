"""Public-output fanout for detached Architecture B run boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .checkpoint_digests import checkpoint_source_projection_digest_manifest
from .engine import _compact_raw_result
from .flat_population_reporting import validated_native_flat_behavior_rows
from .lifecycle_events import (
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
    typed_lifecycle_events_to_rows,
    validated_typed_lifecycle_event_cursor,
    write_typed_lifecycle_public_outputs,
)
from .public_metrics_overlay import (
    NATIVE_LINEAGE_METRICS_INPUT_KEY,
    native_policy_cache_metrics,
    public_metrics_overlay,
)
from .runner_integration import write_native_behavior_public_outputs
from tools.visualization.living_lineage_canopy.python.default_run_export import (
    materialize_default_terminal_canopy,
)


SCHEMA_VERSION = "eiel_architecture_b_public_output_fanout_v1"
RUN_RESULT_SCHEMA_VERSION = "eiel_native_architecture_b_run_result_v1"
BOUNDARY_SCHEMA_VERSION = "eiel_native_architecture_b_boundary_v1"
SNAPSHOT_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "eiel_architecture_b_snapshot_artifact_manifest_v1"
)
SNAPSHOT_COMPLETION_SCHEMA_VERSION = "eiel_architecture_b_snapshot_completion_v1"
SNAPSHOT_STAGING_DIRECTORY = ".architecture_b_snapshots"
SNAPSHOT_ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
SNAPSHOT_COMPLETION_MARKER_NAME = "COMPLETE.json"
SNAPSHOT_FAILURE_RECORD_NAME = "failure.json"
SNAPSHOT_FAILURE_SCHEMA_VERSION = "eiel_architecture_b_snapshot_failure_v1"
METRICS_CONTINUATION_STATE_ATTRIBUTE = (
    "_eiel_architecture_b_metrics_continuation_state"
)
NATIVE_CONTROLLER_PROTOCOL_VERSION = 2
_CHRONICLE_METRIC_FIELDS = (
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "patch_food_amount_mean",
    "patch_food_depleted_frac",
    "occupied_patch_waste_mean",
    "patch_density_hotspot_frac",
    "patch_regrowth_suppressed_total",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_deaths",
    "wdl_delayed_hazard_event_count",
    "wdl_delayed_hazard_damage_total",
    "patch_recovery_positive_total",
)
SNAPSHOT_DIGEST_CONTRACT_SCHEMA_VERSION = (
    "eiel_native_architecture_b_snapshot_digest_contract_v1"
)
SNAPSHOT_DIGEST_DOMAIN = "eiel_native_architecture_b_boundary_snapshot_v2"
SNAPSHOT_DIGEST_EXCLUDED_TOP_LEVEL_FIELDS = (
    "boundary_wall_time_ns",
    "snapshot_digest",
    "snapshot_digest_sha256",
)

_REQUIRED_GENERATION_COUNT_FIELDS = (
    "death_rows",
    "committed_birth_rows",
    "failed_birth_rows",
    "public_failed_birth_rows",
    "birth_attempt_rows",
    "injury_blocked_births",
    "applied_actions",
    "action_move_total",
    "action_move_n",
    "action_move_s",
    "action_move_e",
    "action_move_w",
    "action_gather",
    "action_rest",
    "action_attack",
    "moved",
    "gathered",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "birth_pre_cooldown",
    "birth_pre_no_energy",
    "birth_pre_injury_gate",
    "birth_pre_no_space",
    "birth_pre_no_mate",
    "birth_fail_no_space",
    "birth_fail_no_mate",
    "birth_fail_buffer",
    "birth_fail_injury_gate",
    "birth_fail_scar_gate",
    "birth_fail_conception",
    "birth_fail_gestation",
    "alive_count",
    "capacity",
    "free_slots",
    "typed_event_rows",
)


class ArchitectureBReportingError(RuntimeError):
    """Raised before continuation when detached public output cannot be trusted."""


class ArchitectureBSnapshotPersistenceAcknowledgement(TypedDict):
    """Payload accepted by the native Architecture B persistence mailbox."""

    api_version: int
    request_id: str
    expected_controller_id: str
    expected_owner_epoch: int
    snapshot_id: str
    snapshot_digest_sha256: str
    outcome: str
    artifact_manifest_digest_sha256: str
    acknowledged_channel_cursor_ranges: dict[str, list[int]]
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class _PreparedBoundary:
    index: int
    generation: int
    boundary_wall_time_ns: int
    raw: dict[str, Any]
    behavior_rows: tuple[dict[str, Any], ...]
    counts: dict[str, Any]
    overlay: dict[str, Any]
    cohort_size: int
    cohort_survivors: int
    lifecycle_events: dict[str, Any]
    lifecycle_start_cursor: int
    lifecycle_end_cursor: int


@dataclass(frozen=True, slots=True)
class _ResolvedPersistenceSchedule:
    explicit: bool
    due_compact_progress: bool
    due_exact_reporting: bool
    due_chronicle: bool
    due_checkpoint: bool
    schedule_state: dict[str, Any]
    terminal_candidate: str
    cursor_before: dict[str, Any]
    cursor_after: dict[str, Any]
    payload_byte_counts: dict[str, Any]
    channel_cursor_ranges: dict[str, list[int]]
    acknowledgement_cursor_ranges: dict[str, list[int]]
    due_facets: tuple[str, ...]


def _fail(message: str) -> ArchitectureBReportingError:
    return ArchitectureBReportingError(
        f"EIEL-ARCHITECTURE-B-REPORTING-FAILED: {message}"
    )


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{path} must be a mapping")
    return dict(value)


def _sequence(value: Any, *, path: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _fail(f"{path} must be a sequence")
    return list(value)


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], *, path: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise _fail(f"{path} is missing required fields: {missing}")


def _strict_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise _fail(f"{path} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise _fail(f"{path} must be at least {minimum}")
    return resolved


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{path} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise _fail(f"{path} must be finite")
    return resolved


def _required_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{path} must be a non-empty string")
    return value


def _lowercase_sha256(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise _fail(f"{path} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_bool(
    value: Mapping[str, Any],
    field: str,
    expected: bool,
    *,
    path: str,
) -> None:
    if field not in value or type(value[field]) is not bool or value[field] is not expected:
        raise _fail(f"{path}.{field} must be {expected}")


def _count(raw: Mapping[str, Any], field: str) -> int:
    if field not in raw:
        raise _fail(f"generation_result.{field} is required")
    return _strict_int(raw[field], path=f"generation_result.{field}")


def _public_output_counts(
    raw: Mapping[str, Any],
    compact: Mapping[str, Any],
) -> dict[str, Any]:
    _require_fields(raw, _REQUIRED_GENERATION_COUNT_FIELDS, path="generation_result")
    values = {
        "schema_version": "eiel_native_runtime_public_output_counts_v1",
        "source": "native_compact_result",
        "deaths": _count(raw, "death_rows"),
        "successful_births": _count(raw, "committed_birth_rows"),
        "failed_births": _count(raw, "public_failed_birth_rows"),
        "native_failed_birth_event_rows": _count(raw, "failed_birth_rows"),
        "birth_attempts": _count(raw, "birth_attempt_rows"),
        "injury_blocked_births": _count(raw, "injury_blocked_births"),
        "behavior_action_total": _count(raw, "applied_actions"),
        "behavior_move_total": _count(raw, "action_move_total"),
        "behavior_move_n_total": _count(raw, "action_move_n"),
        "behavior_move_s_total": _count(raw, "action_move_s"),
        "behavior_move_e_total": _count(raw, "action_move_e"),
        "behavior_move_w_total": _count(raw, "action_move_w"),
        "behavior_gather_total": _count(raw, "action_gather"),
        "behavior_rest_total": _count(raw, "action_rest"),
        "behavior_attack_total": _count(raw, "action_attack"),
        "native_moved_successfully": _count(raw, "moved"),
        "native_gathered_successfully": _count(raw, "gathered"),
        "native_movement_attempts": _count(raw, "movement_attempts"),
        "native_movement_blocked_occupied": _count(
            raw, "movement_blocked_occupied"
        ),
        "native_movement_blocked_bounds": _count(raw, "movement_blocked_bounds"),
        "birth_pre_cooldown": _count(raw, "birth_pre_cooldown"),
        "birth_pre_no_energy": _count(raw, "birth_pre_no_energy"),
        "birth_pre_injury_gate": _count(raw, "birth_pre_injury_gate"),
        "birth_pre_no_space": _count(raw, "birth_pre_no_space"),
        "birth_pre_no_mate": _count(raw, "birth_pre_no_mate"),
        "birth_fail_no_space": _count(raw, "birth_fail_no_space"),
        "birth_fail_no_mate": _count(raw, "birth_fail_no_mate"),
        "birth_fail_buffer": _count(raw, "birth_fail_buffer"),
        "birth_fail_injury_gate": _count(raw, "birth_fail_injury_gate"),
        "birth_fail_scar_gate": _count(raw, "birth_fail_scar_gate"),
        "birth_fail_conception": _count(raw, "birth_fail_conception"),
        "birth_fail_gestation": _count(raw, "birth_fail_gestation"),
        "alive": _count(raw, "alive_count"),
        "capacity": _count(raw, "capacity"),
        "free_slots": _count(raw, "free_slots"),
    }
    values.update(native_policy_cache_metrics(raw, compact))
    return values


def _projection_tensor_bytes(projection: Mapping[str, Any], *, path: str) -> int:
    tensors = _mapping(projection.get("tensors"), path=f"{path}.tensors")
    manifest = _mapping(
        projection.get("tensor_manifest"), path=f"{path}.tensor_manifest"
    )
    if set(tensors) != set(manifest):
        raise _fail(f"{path}.tensor_manifest fields do not match tensors")
    total = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        row = _mapping(manifest[name], path=f"{path}.tensor_manifest.{name}")
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            raise _fail(f"{path}.tensors.{name} is not tensor-like")
        exact = int(tensor.numel()) * int(tensor.element_size())
        recorded = _strict_int(
            row.get("bytes"), path=f"{path}.tensor_manifest.{name}.bytes"
        )
        if recorded != exact:
            raise _fail(
                f"{path}.tensor_manifest.{name}.bytes mismatch: "
                f"recorded={recorded} exact={exact}"
            )
        total += exact
    return total


def _validate_projection_digest(
    projection: Mapping[str, Any],
    recorded_manifests: Any,
    *,
    path: str,
) -> None:
    manifests = _mapping(recorded_manifests, path=f"{path}_digest_manifests")
    source = _mapping(
        manifests.get("source_projection"),
        path=f"{path}_digest_manifests.source_projection",
    )
    digest = _required_string(
        source.get("digest"), path=f"{path}_digest_manifests.source_projection.digest"
    )
    actual = checkpoint_source_projection_digest_manifest(projection)["digest"]
    if digest != actual:
        raise _fail(f"{path} source-projection digest mismatch")


def _validate_projection_boundary(
    projection: Mapping[str, Any],
    *,
    path: str,
    next_generation: int,
) -> Mapping[str, Any]:
    _required_string(projection.get("schema_version"), path=f"{path}.schema_version")
    metadata = _mapping(projection.get("metadata"), path=f"{path}.metadata")
    if (
        _strict_int(metadata.get("generation"), path=f"{path}.metadata.generation")
        != next_generation
    ):
        raise _fail(f"{path}.metadata.generation does not match next_generation")
    if _strict_int(metadata.get("tick"), path=f"{path}.metadata.tick") != 0:
        raise _fail(f"{path}.metadata.tick must be zero")
    if metadata.get("agent_grid_runner_phase") != "generation_boundary":
        raise _fail(f"{path}.metadata.agent_grid_runner_phase is not generation_boundary")
    return metadata


def _validate_sequence_tensor(events: Mapping[str, Any], *, path: str) -> None:
    sequence = events.get("sequence")
    if sequence is None or not hasattr(sequence, "detach"):
        raise _fail(f"{path}.sequence is missing")
    actual = [int(value) for value in sequence.detach().cpu().tolist()]
    if actual != list(range(len(actual))):
        raise _fail(f"{path}.sequence must be contiguous from zero")


def _validate_result_header(
    result: Mapping[str, Any], boundary_count: int
) -> tuple[str, str, int, int, int]:
    required = (
        "schema_version",
        "terminal_status",
        "run_id",
        "controller_id",
        "owner_epoch",
        "executed",
        "native_owner",
        "hidden_fallback",
        "runtime_jit",
        "success_marker_allowed",
        "start_generation",
        "requested_generations",
        "completed_generations",
        "next_generation",
        "next_tick",
        "boundary_snapshot_count",
        "final_snapshot_sequence",
        "trajectory_dispatch_count",
        "python_generation_dispatches",
        "projection_export_count",
        "boundary_tensor_manifest_bytes",
        "final_checkpoint_projection",
        "final_checkpoint_projection_digest_manifests",
        "snapshot_validation_receipt",
    )
    _require_fields(result, required, path="run_result")
    if result["schema_version"] != RUN_RESULT_SCHEMA_VERSION:
        raise _fail("run_result schema mismatch")
    if result["terminal_status"] != "run_complete":
        raise _fail("run_result must be terminal run_complete")
    for field, expected in (
        ("executed", True),
        ("native_owner", True),
        ("hidden_fallback", False),
        ("runtime_jit", False),
        ("success_marker_allowed", True),
    ):
        _require_bool(result, field, expected, path="run_result")
    run_id = _required_string(result["run_id"], path="run_result.run_id")
    controller_id = _required_string(
        result["controller_id"], path="run_result.controller_id"
    )
    owner_epoch = _strict_int(
        result["owner_epoch"], path="run_result.owner_epoch", minimum=1
    )
    start_generation = _strict_int(
        result["start_generation"], path="run_result.start_generation"
    )
    final_snapshot_sequence = _strict_int(
        result["final_snapshot_sequence"],
        path="run_result.final_snapshot_sequence",
        minimum=1,
    )
    initial_snapshot_sequence = final_snapshot_sequence - boundary_count
    if initial_snapshot_sequence < 0:
        raise _fail(
            "run_result.final_snapshot_sequence precedes the returned boundaries"
        )
    exact_scalars = {
        "requested_generations": boundary_count,
        "completed_generations": boundary_count,
        "next_generation": start_generation + boundary_count,
        "next_tick": 0,
        "boundary_snapshot_count": boundary_count,
        "trajectory_dispatch_count": 1,
        "python_generation_dispatches": 0,
        "projection_export_count": boundary_count * 2,
    }
    for field, expected in exact_scalars.items():
        actual = _strict_int(result[field], path=f"run_result.{field}")
        if actual != expected:
            raise _fail(
                f"run_result.{field} mismatch: expected={expected} actual={actual}"
            )
    receipt = _mapping(
        result["snapshot_validation_receipt"],
        path="run_result.snapshot_validation_receipt",
    )
    _require_fields(
        receipt,
        (
            "schema_version",
            "run_id",
            "controller_id",
            "owner_epoch",
            "acknowledged",
            "disposition",
            "terminal_status",
            "snapshot_frontier",
            "final_snapshot_sequence",
            "validated_source_projection_digest",
            "success_marker_allowed",
        ),
        path="run_result.snapshot_validation_receipt",
    )
    if receipt["schema_version"] != (
        "eiel_native_architecture_b_snapshot_validation_receipt_v1"
    ):
        raise _fail("run_result.snapshot_validation_receipt schema mismatch")
    if receipt["run_id"] != run_id or receipt["controller_id"] != controller_id:
        raise _fail("run_result.snapshot_validation_receipt identity mismatch")
    if _strict_int(
        receipt["owner_epoch"],
        path="run_result.snapshot_validation_receipt.owner_epoch",
        minimum=1,
    ) != owner_epoch:
        raise _fail("run_result.snapshot_validation_receipt owner epoch mismatch")
    _require_bool(
        receipt,
        "acknowledged",
        True,
        path="run_result.snapshot_validation_receipt",
    )
    _require_bool(
        receipt,
        "success_marker_allowed",
        True,
        path="run_result.snapshot_validation_receipt",
    )
    if receipt["disposition"] != "accepted" or receipt["terminal_status"] != "run_complete":
        raise _fail("run_result.snapshot_validation_receipt was not accepted")
    final_manifests = _mapping(
        result["final_checkpoint_projection_digest_manifests"],
        path="run_result.final_checkpoint_projection_digest_manifests",
    )
    final_source_manifest = _mapping(
        final_manifests.get("source_projection"),
        path=(
            "run_result.final_checkpoint_projection_digest_manifests."
            "source_projection"
        ),
    )
    final_source_digest = _lowercase_sha256(
        final_source_manifest.get("digest"),
        path=(
            "run_result.final_checkpoint_projection_digest_manifests."
            "source_projection.digest"
        ),
    )
    receipt_source_digest = _lowercase_sha256(
        receipt["validated_source_projection_digest"],
        path=(
            "run_result.snapshot_validation_receipt."
            "validated_source_projection_digest"
        ),
    )
    if receipt_source_digest != final_source_digest:
        raise _fail(
            "run_result.snapshot_validation_receipt."
            "validated_source_projection_digest does not match "
            "run_result.final_checkpoint_projection_digest_manifests."
            "source_projection.digest"
        )
    if _strict_int(
        receipt["snapshot_frontier"],
        path="run_result.snapshot_validation_receipt.snapshot_frontier",
        minimum=-1,
    ) != start_generation + boundary_count - 1:
        raise _fail("run_result.snapshot_validation_receipt frontier mismatch")
    if _strict_int(
        receipt["final_snapshot_sequence"],
        path="run_result.snapshot_validation_receipt.final_snapshot_sequence",
        minimum=1,
    ) != final_snapshot_sequence:
        raise _fail("run_result.snapshot_validation_receipt sequence mismatch")
    return (
        run_id,
        controller_id,
        owner_epoch,
        start_generation,
        initial_snapshot_sequence,
    )


def _prepare_boundary(
    runner: Any,
    item: Any,
    *,
    index: int,
    run_id: str,
    controller_id: str,
    owner_epoch: int,
    expected_generation: int,
    expected_snapshot_sequence: int,
    previous_event_cursor: int | None,
) -> tuple[_PreparedBoundary, int, int]:
    path = f"run_result.boundaries[{index}]"
    boundary = _mapping(item, path=path)
    required = (
        "schema_version",
        "run_id",
        "controller_id",
        "owner_epoch",
        "snapshot_id",
        "snapshot_sequence",
        "completed_generation",
        "next_generation",
        "tick",
        "phase",
        "controller_status_at_clone",
        "generation_end_reporting_frame_role",
        "generation_result",
        "turnover_result",
        "generation_end_reporting_frame",
        "post_turnover_checkpoint_projection",
        "generation_end_reporting_frame_tensor_bytes",
        "post_turnover_projection_tensor_bytes",
        "reporting_projection_digest_manifests",
        "checkpoint_projection_digest_manifests",
        "tensor_manifest_boundary_bytes",
        "boundary_wall_time_ns",
    )
    _require_fields(boundary, required, path=path)
    if boundary["schema_version"] != BOUNDARY_SCHEMA_VERSION:
        raise _fail(f"{path}.schema_version mismatch")
    if boundary["run_id"] != run_id:
        raise _fail(f"{path}.run_id mismatch")
    if boundary["controller_id"] != controller_id:
        raise _fail(f"{path}.controller_id mismatch")
    if _strict_int(boundary["owner_epoch"], path=f"{path}.owner_epoch") != owner_epoch:
        raise _fail(f"{path}.owner_epoch mismatch")
    snapshot_id = _required_string(
        boundary["snapshot_id"], path=f"{path}.snapshot_id"
    )
    expected_scalars = {
        "snapshot_sequence": expected_snapshot_sequence,
        "completed_generation": expected_generation,
        "next_generation": expected_generation + 1,
        "tick": 0,
    }
    for field, expected in expected_scalars.items():
        actual = _strict_int(boundary[field], path=f"{path}.{field}")
        if actual != expected:
            raise _fail(f"{path}.{field} mismatch: expected={expected} actual={actual}")
    expected_snapshot_id = (
        f"{controller_id}:owner:{owner_epoch}:boundary:{expected_snapshot_sequence}"
    )
    if snapshot_id != expected_snapshot_id:
        raise _fail(f"{path}.snapshot_id mismatch")
    if boundary["phase"] != "generation_complete":
        raise _fail(f"{path}.phase must be generation_complete")
    if boundary["controller_status_at_clone"] not in {
        "generation_complete_waiting_persistence",
        "generation_complete_ready_to_resolve_terminal",
    }:
        raise _fail(f"{path}.controller_status_at_clone mismatch")
    if boundary["generation_end_reporting_frame_role"] != (
        "transaction_local_generation_end_values_folded_after_turnover_commit"
    ):
        raise _fail(f"{path}.generation_end_reporting_frame_role mismatch")

    raw = _mapping(boundary["generation_result"], path=f"{path}.generation_result")
    _require_fields(
        raw,
        (
            "schema_version",
            "executed",
            "hidden_fallback",
            "native_owner",
            "single_native_generation_call",
            "generation_index",
            "steps_per_generation",
            "native_tick_loop_count",
            "behavior_rows",
            "peak_injury",
            *_REQUIRED_GENERATION_COUNT_FIELDS,
        ),
        path=f"{path}.generation_result",
    )
    _required_string(raw["schema_version"], path=f"{path}.generation_result.schema_version")
    for field, expected in (
        ("executed", True),
        ("hidden_fallback", False),
        ("native_owner", True),
        ("single_native_generation_call", True),
    ):
        _require_bool(raw, field, expected, path=f"{path}.generation_result")
    generation = _strict_int(
        raw["generation_index"], path=f"{path}.generation_result.generation_index"
    )
    if generation != expected_generation:
        raise _fail(f"{path}.generation_result.generation_index mismatch")
    steps = _strict_int(
        raw["steps_per_generation"],
        path=f"{path}.generation_result.steps_per_generation",
        minimum=1,
    )
    if _strict_int(
        raw["native_tick_loop_count"],
        path=f"{path}.generation_result.native_tick_loop_count",
        minimum=1,
    ) != steps:
        raise _fail(f"{path}.generation_result native tick count mismatch")
    peak_injury = _finite_float(
        raw["peak_injury"], path=f"{path}.generation_result.peak_injury"
    )
    boundary_wall_time_ns = _strict_int(
        boundary["boundary_wall_time_ns"],
        path=f"{path}.boundary_wall_time_ns",
    )

    behavior_rows = validated_native_flat_behavior_rows(
        raw["behavior_rows"], expected_generation=generation
    )
    compact = _compact_raw_result(raw)
    counts = _public_output_counts(raw, compact)
    directional_moves = sum(
        counts[name]
        for name in (
            "behavior_move_n_total",
            "behavior_move_s_total",
            "behavior_move_e_total",
            "behavior_move_w_total",
        )
    )
    if counts["behavior_move_total"] != directional_moves:
        raise _fail(f"{path}.generation_result action_move_total is inconsistent")
    action_total = counts["behavior_move_total"] + sum(
        counts[name]
        for name in (
            "behavior_gather_total",
            "behavior_rest_total",
            "behavior_attack_total",
        )
    )
    if counts["behavior_action_total"] != action_total:
        raise _fail(f"{path}.generation_result applied_actions is inconsistent")
    if counts["capacity"] != counts["alive"] + counts["free_slots"]:
        raise _fail(f"{path}.generation_result capacity counters are inconsistent")
    if sum(int(row["alive_end"]) for row in behavior_rows) != counts["alive"]:
        raise _fail(f"{path}.generation_result behavior alive rows are inconsistent")

    reporting = _mapping(
        boundary["generation_end_reporting_frame"],
        path=f"{path}.generation_end_reporting_frame",
    )
    post = _mapping(
        boundary["post_turnover_checkpoint_projection"],
        path=f"{path}.post_turnover_checkpoint_projection",
    )
    reporting_metadata = _validate_projection_boundary(
        reporting,
        path=f"{path}.generation_end_reporting_frame",
        next_generation=generation + 1,
    )
    post_metadata = _validate_projection_boundary(
        post,
        path=f"{path}.post_turnover_checkpoint_projection",
        next_generation=generation + 1,
    )
    reporting_bytes = _projection_tensor_bytes(
        reporting, path=f"{path}.generation_end_reporting_frame"
    )
    post_bytes = _projection_tensor_bytes(
        post, path=f"{path}.post_turnover_checkpoint_projection"
    )
    if _strict_int(
        boundary["generation_end_reporting_frame_tensor_bytes"],
        path=f"{path}.generation_end_reporting_frame_tensor_bytes",
    ) != reporting_bytes:
        raise _fail(f"{path}.generation_end_reporting_frame_tensor_bytes mismatch")
    if _strict_int(
        boundary["post_turnover_projection_tensor_bytes"],
        path=f"{path}.post_turnover_projection_tensor_bytes",
    ) != post_bytes:
        raise _fail(f"{path}.post_turnover_projection_tensor_bytes mismatch")
    byte_manifest = _mapping(
        boundary["tensor_manifest_boundary_bytes"],
        path=f"{path}.tensor_manifest_boundary_bytes",
    )
    expected_bytes = {
        "generation_end_reporting_frame": reporting_bytes,
        "post_turnover_checkpoint_projection": post_bytes,
        "total": reporting_bytes + post_bytes,
    }
    for field, expected in expected_bytes.items():
        actual = _strict_int(
            byte_manifest.get(field),
            path=f"{path}.tensor_manifest_boundary_bytes.{field}",
        )
        if actual != expected:
            raise _fail(f"{path}.tensor_manifest_boundary_bytes.{field} mismatch")
    _validate_projection_digest(
        reporting,
        boundary["reporting_projection_digest_manifests"],
        path=f"{path}.reporting_projection",
    )
    _validate_projection_digest(
        post,
        boundary["checkpoint_projection_digest_manifests"],
        path=f"{path}.checkpoint_projection",
    )
    if _strict_int(
        reporting_metadata.get("alive_count"),
        path=f"{path}.generation_end_reporting_frame.metadata.alive_count",
    ) != counts["alive"]:
        raise _fail(f"{path} reporting-frame alive count mismatch")

    turnover = _mapping(boundary["turnover_result"], path=f"{path}.turnover_result")
    _require_fields(
        turnover,
        (
            "executed",
            "hidden_fallback",
            "native_owner",
            "next_generation",
            "next_tick",
            "alive_before",
            "alive_after",
            "population_size",
            "typed_event_rows",
        ),
        path=f"{path}.turnover_result",
    )
    for field, expected in (
        ("executed", True),
        ("hidden_fallback", False),
        ("native_owner", True),
    ):
        _require_bool(turnover, field, expected, path=f"{path}.turnover_result")
    if (
        _strict_int(
            turnover["next_generation"],
            path=f"{path}.turnover_result.next_generation",
        )
        != generation + 1
    ):
        raise _fail(f"{path}.turnover_result.next_generation mismatch")
    if _strict_int(turnover["next_tick"], path=f"{path}.turnover_result.next_tick") != 0:
        raise _fail(f"{path}.turnover_result.next_tick must be zero")
    if (
        _strict_int(
            turnover["alive_before"], path=f"{path}.turnover_result.alive_before"
        )
        != counts["alive"]
    ):
        raise _fail(f"{path}.turnover_result.alive_before mismatch")
    alive_after = _strict_int(
        turnover["alive_after"], path=f"{path}.turnover_result.alive_after"
    )
    if (
        _strict_int(
            turnover["population_size"],
            path=f"{path}.turnover_result.population_size",
        )
        != alive_after
    ):
        raise _fail(f"{path}.turnover_result.population_size mismatch")
    if (
        _strict_int(
            post_metadata.get("alive_count"),
            path=(
                f"{path}.post_turnover_checkpoint_projection.metadata.alive_count"
            ),
        )
        != alive_after
    ):
        raise _fail(f"{path} post-turnover alive count mismatch")

    reporting_events = typed_lifecycle_events_from_projection(reporting)
    reporting_cursor = validated_typed_lifecycle_event_cursor(
        reporting, reporting_events
    )
    lifecycle_events = typed_lifecycle_events_from_projection(post)
    lifecycle_end_cursor = validated_typed_lifecycle_event_cursor(post, lifecycle_events)
    _validate_sequence_tensor(
        reporting_events, path=f"{path}.generation_end_reporting_frame.lifecycle"
    )
    _validate_sequence_tensor(
        lifecycle_events, path=f"{path}.post_turnover_checkpoint_projection.lifecycle"
    )
    generation_event_rows = _count(raw, "typed_event_rows")
    turnover_event_rows = _strict_int(
        turnover["typed_event_rows"], path=f"{path}.turnover_result.typed_event_rows"
    )
    lifecycle_start_cursor = reporting_cursor - generation_event_rows
    if lifecycle_start_cursor < 0:
        raise _fail(f"{path} generation lifecycle delta exceeds reporting cursor")
    if previous_event_cursor is not None and lifecycle_start_cursor != previous_event_cursor:
        raise _fail(f"{path} lifecycle cursor does not continue the previous boundary")
    if lifecycle_end_cursor - reporting_cursor != turnover_event_rows:
        raise _fail(f"{path} turnover lifecycle delta is inconsistent")

    overlay_values = public_metrics_overlay(
        raw,
        compact,
        reporting,
        sim_cfg=getattr(runner, "sim_cfg", None),
        env_cfg=getattr(runner, "env_cfg", None),
        metrics_generation=generation,
    )
    if NATIVE_LINEAGE_METRICS_INPUT_KEY not in overlay_values:
        raise _fail(f"{path} did not produce native lineage metrics input")
    cohort_size = sum(row["cohort"] == "start" for row in behavior_rows)
    cohort_survivors = sum(
        row["cohort"] == "start" and int(row["alive_end"]) == 1
        for row in behavior_rows
    )
    counts["peak_injury"] = peak_injury
    return (
        _PreparedBoundary(
            index=index,
            generation=generation,
            boundary_wall_time_ns=boundary_wall_time_ns,
            raw=raw,
            behavior_rows=behavior_rows,
            counts=counts,
            overlay={
                "schema_version": "eiel_native_public_metrics_overlay_v1",
                "source": "native_owner_checkpoint_projection",
                "values": dict(overlay_values),
            },
            cohort_size=int(cohort_size),
            cohort_survivors=int(cohort_survivors),
            lifecycle_events=lifecycle_events,
            lifecycle_start_cursor=lifecycle_start_cursor,
            lifecycle_end_cursor=lifecycle_end_cursor,
        ),
        lifecycle_end_cursor,
        reporting_bytes + post_bytes,
    )


def _prepare_fanout(
    runner: Any,
    run_result: Mapping[str, Any],
) -> tuple[list[_PreparedBoundary], Path]:
    result = _mapping(run_result, path="run_result")
    boundaries = _sequence(result.get("boundaries"), path="run_result.boundaries")
    if not boundaries:
        raise _fail("run_result.boundaries must not be empty")
    (
        run_id,
        controller_id,
        owner_epoch,
        start_generation,
        initial_snapshot_sequence,
    ) = _validate_result_header(result, len(boundaries))

    logger = getattr(runner, "logger", None)
    if logger is None or not callable(getattr(logger, "log_generation", None)):
        raise _fail("runner.logger.log_generation is required")
    sim_cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if sim_cfg is None or env_cfg is None:
        raise _fail("runner.sim_cfg and runner.env_cfg are required")
    if not bool(getattr(sim_cfg, "behavior_log_enable", True)):
        raise _fail("Architecture B exact reporting requires behavior_log_enable")
    if not bool(getattr(sim_cfg, "lineage_observability_enable", True)):
        raise _fail("Architecture B exact reporting requires lineage_observability_enable")
    run_dir_value = str(getattr(runner, "run_dir", "") or "")
    if not run_dir_value:
        raise _fail("runner.run_dir is required")
    run_dir = Path(run_dir_value)

    prepared: list[_PreparedBoundary] = []
    seen_snapshot_ids: set[str] = set()
    previous_cursor: int | None = None
    total_boundary_bytes = 0
    for index, item in enumerate(boundaries):
        try:
            boundary, previous_cursor, boundary_bytes = _prepare_boundary(
                runner,
                item,
                index=index,
                run_id=run_id,
                controller_id=controller_id,
                owner_epoch=owner_epoch,
                expected_generation=start_generation + index,
                expected_snapshot_sequence=initial_snapshot_sequence + index + 1,
                previous_event_cursor=previous_cursor,
            )
        except ArchitectureBReportingError:
            raise
        except Exception as exc:
            raise _fail(f"run_result.boundaries[{index}] validation failed: {exc}") from exc
        snapshot_id = str(_mapping(item, path=f"run_result.boundaries[{index}]")["snapshot_id"])
        if snapshot_id in seen_snapshot_ids:
            raise _fail(f"run_result.boundaries[{index}].snapshot_id is duplicated")
        seen_snapshot_ids.add(snapshot_id)
        total_boundary_bytes += boundary_bytes
        prepared.append(boundary)

    recorded_total = _strict_int(
        result["boundary_tensor_manifest_bytes"],
        path="run_result.boundary_tensor_manifest_bytes",
    )
    if recorded_total != total_boundary_bytes:
        raise _fail(
            "run_result.boundary_tensor_manifest_bytes mismatch: "
            f"recorded={recorded_total} exact={total_boundary_bytes}"
        )
    final_projection = _mapping(
        result["final_checkpoint_projection"], path="run_result.final_checkpoint_projection"
    )
    _validate_projection_digest(
        final_projection,
        result["final_checkpoint_projection_digest_manifests"],
        path="run_result.final_checkpoint_projection",
    )
    final_digest = checkpoint_source_projection_digest_manifest(final_projection)["digest"]
    last_boundary = _mapping(boundaries[-1], path="run_result.boundaries[-1]")
    last_post = _mapping(
        last_boundary["post_turnover_checkpoint_projection"],
        path="run_result.boundaries[-1].post_turnover_checkpoint_projection",
    )
    if checkpoint_source_projection_digest_manifest(last_post)["digest"] != final_digest:
        raise _fail("run_result.final_checkpoint_projection does not match the last boundary")
    return prepared, run_dir


def _canonical_json_bytes(value: Any, *, path: str) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"{path} is not canonically JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_digest_length(digest: Any, value: int, *, path: str) -> None:
    if value < 0 or value > (1 << 64) - 1:
        raise _fail(f"{path} is outside the canonical uint64 range")
    digest.update(value.to_bytes(8, byteorder="big", signed=False))


def _snapshot_digest_text(
    digest: Any,
    tag: bytes,
    value: str | bytes,
    *,
    path: str,
) -> None:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(tag)
    _snapshot_digest_length(digest, len(payload), path=f"{path}.length")
    digest.update(payload)


def _snapshot_digest_float64(digest: Any, value: float) -> None:
    if math.isnan(value):
        digest.update(bytes.fromhex("7ff8000000000000"))
    elif value == 0.0:
        digest.update(b"\x00" * 8)
    else:
        digest.update(struct.pack(">d", value))


_TORCH_SCALAR_TYPE_NAMES = {
    "torch.uint8": "Byte",
    "torch.uint16": "UInt16",
    "torch.uint32": "UInt32",
    "torch.uint64": "UInt64",
    "torch.int8": "Char",
    "torch.int16": "Short",
    "torch.int32": "Int",
    "torch.int64": "Long",
    "torch.float16": "Half",
    "torch.float32": "Float",
    "torch.float64": "Double",
    "torch.bool": "Bool",
    "torch.bfloat16": "BFloat16",
    "torch.float8_e4m3fn": "Float8_e4m3fn",
    "torch.float8_e4m3fnuz": "Float8_e4m3fnuz",
    "torch.float8_e5m2": "Float8_e5m2",
    "torch.float8_e5m2fnuz": "Float8_e5m2fnuz",
}


def _update_snapshot_tensor_digest(digest: Any, value: Any, *, path: str) -> None:
    try:
        tensor = value.detach().cpu().contiguous()
        dtype = _TORCH_SCALAR_TYPE_NAMES.get(str(tensor.dtype))
        if dtype is None:
            raise _fail(
                f"{path} has unsupported canonical tensor dtype {tensor.dtype}"
            )
        _snapshot_digest_text(digest, b"q", dtype, path=f"{path}.dtype")
        shape = [int(size) for size in tensor.shape]
        _snapshot_digest_length(digest, len(shape), path=f"{path}.rank")
        for index, size in enumerate(shape):
            _snapshot_digest_length(
                digest, size, path=f"{path}.shape[{index}]"
            )
        numel = int(tensor.numel())
        if dtype == "Bool":
            _snapshot_digest_length(digest, numel, path=f"{path}.bytes")
            digest.update(
                bytes(1 if bool(item) else 0 for item in tensor.reshape(-1).tolist())
            )
            return
        _snapshot_digest_length(digest, numel * 8, path=f"{path}.bytes")
        if bool(tensor.is_floating_point()):
            for item in tensor.double().reshape(-1).tolist():
                _snapshot_digest_float64(digest, float(item))
            return
        for item in tensor.long().reshape(-1).tolist():
            _snapshot_digest_length(
                digest,
                int(item) & ((1 << 64) - 1),
                path=f"{path}.value",
            )
    except ArchitectureBReportingError:
        raise
    except Exception as exc:
        raise _fail(f"{path} tensor content could not be hashed: {exc}") from exc


def _update_snapshot_value_digest(digest: Any, value: Any, *, path: str) -> None:
    if value is None:
        digest.update(b"n")
        return
    if isinstance(value, bool):
        digest.update(b"T" if value else b"F")
        return
    if isinstance(value, int):
        _snapshot_digest_text(digest, b"i", str(value), path=path)
        return
    if isinstance(value, float):
        digest.update(b"f")
        _snapshot_digest_float64(digest, value)
        return
    if isinstance(value, str):
        _snapshot_digest_text(digest, b"s", value, path=path)
        return
    if isinstance(value, bytes):
        _snapshot_digest_text(digest, b"b", value, path=path)
        return
    if isinstance(value, dict):
        digest.update(b"d")
        _snapshot_digest_length(digest, len(value), path=f"{path}.length")
        items: list[tuple[bytes, str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(f"{path} contains a non-string mapping key")
            items.append((key.encode("utf-8"), key, item))
        for _encoded_key, key, item in sorted(items, key=lambda row: row[0]):
            _snapshot_digest_text(digest, b"k", key, path=f"{path}.{key}.key")
            _update_snapshot_value_digest(digest, item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"t" if isinstance(value, tuple) else b"l")
        _snapshot_digest_length(digest, len(value), path=f"{path}.length")
        for index, item in enumerate(value):
            _update_snapshot_value_digest(
                digest, item, path=f"{path}[{index}]"
            )
        return
    if all(hasattr(value, name) for name in ("detach", "cpu", "contiguous")):
        _update_snapshot_tensor_digest(digest, value, path=path)
        return
    raise _fail(
        f"{path} contains unsupported canonical value type {type(value).__name__}"
    )


def _expected_snapshot_digest_contract(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    non_string_keys = [key for key in snapshot if not isinstance(key, str)]
    if non_string_keys:
        raise _fail("snapshot canonical digest contract requires string keys")
    covered_fields = {
        key
        for key in snapshot
        if key not in SNAPSHOT_DIGEST_EXCLUDED_TOP_LEVEL_FIELDS
    }
    covered_fields.add("snapshot_digest_contract")
    return {
        "schema_version": SNAPSHOT_DIGEST_CONTRACT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "domain": SNAPSHOT_DIGEST_DOMAIN,
        "covered_scope": "all_listed_top_level_fields_recursively",
        "text_encoding": "utf8",
        "mapping_order": "ascending_unsigned_utf8_key_bytes",
        "length_encoding": "unsigned_64_bit_big_endian",
        "integer_encoding": "ascii_decimal_tagged_and_length_prefixed",
        "float_encoding": (
            "ieee754_binary64_big_endian_canonical_quiet_nan_signed_zero_collapsed"
        ),
        "tensor_encoding": (
            "dtype_name_shape_then_bool_bytes_or_canonical_float64_or_"
            "twos_complement_int64"
        ),
        "sequence_encoding": "list_or_tuple_tag_length_then_recursive_values",
        "excluded_top_level_fields": list(
            SNAPSHOT_DIGEST_EXCLUDED_TOP_LEVEL_FIELDS
        ),
        "covered_top_level_fields": sorted(
            covered_fields, key=lambda field: field.encode("utf-8")
        ),
    }


def _boundary_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in SNAPSHOT_DIGEST_EXCLUDED_TOP_LEVEL_FIELDS
    }
    digest = hashlib.sha256()
    _snapshot_digest_text(digest, b"D", SNAPSHOT_DIGEST_DOMAIN, path="domain")
    _update_snapshot_value_digest(digest, payload, path="snapshot")
    return digest.hexdigest()


def _validated_snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    if (
        _strict_int(
            snapshot.get("controller_protocol_version"),
            path="snapshot.controller_protocol_version",
        )
        != NATIVE_CONTROLLER_PROTOCOL_VERSION
    ):
        raise _fail("snapshot controller protocol does not advertise Phase 4 digest truth")
    contract = _mapping(
        snapshot.get("snapshot_digest_contract"),
        path="snapshot.snapshot_digest_contract",
    )
    expected_contract = _expected_snapshot_digest_contract(snapshot)
    if contract != expected_contract:
        raise _fail("snapshot canonical digest contract does not match the native protocol")
    declared = _lowercase_sha256(
        snapshot.get("snapshot_digest_sha256"),
        path="snapshot.snapshot_digest_sha256",
    )
    if (
        _lowercase_sha256(
            snapshot.get("snapshot_digest"), path="snapshot.snapshot_digest"
        )
        != declared
    ):
        raise _fail("snapshot digest fields disagree")
    computed = _boundary_snapshot_sha256(snapshot)
    if computed != declared:
        raise _fail(
            "snapshot canonical digest mismatch: native digest was not independently verified"
        )
    return computed


def _validated_mailbox_snapshot_header(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    boundary = _mapping(snapshot, path="snapshot")
    _require_fields(
        boundary,
        (
            "schema_version",
            "run_id",
            "controller_id",
            "owner_epoch",
            "snapshot_id",
            "snapshot_sequence",
            "boundary_ordinal",
            "completed_generation",
            "next_generation",
            "tick",
            "phase",
            "controller_status_at_clone",
            "controller_protocol_version",
            "snapshot_digest_contract",
            "snapshot_digest",
            "snapshot_digest_sha256",
            "snapshot_digest_pending_adapter",
            "persistence_ack_required",
        ),
        path="snapshot",
    )
    if boundary["schema_version"] != BOUNDARY_SCHEMA_VERSION:
        raise _fail("snapshot.schema_version mismatch")
    run_id = _required_string(boundary["run_id"], path="snapshot.run_id")
    controller_id = _required_string(
        boundary["controller_id"], path="snapshot.controller_id"
    )
    owner_epoch = _strict_int(
        boundary["owner_epoch"], path="snapshot.owner_epoch", minimum=1
    )
    sequence = _strict_int(
        boundary["snapshot_sequence"], path="snapshot.snapshot_sequence", minimum=1
    )
    ordinal = _strict_int(
        boundary["boundary_ordinal"], path="snapshot.boundary_ordinal", minimum=1
    )
    if sequence != ordinal:
        raise _fail("snapshot.boundary_ordinal does not match snapshot_sequence")
    generation = _strict_int(
        boundary["completed_generation"], path="snapshot.completed_generation"
    )
    if (
        _strict_int(boundary["next_generation"], path="snapshot.next_generation")
        != generation + 1
    ):
        raise _fail("snapshot.next_generation does not follow completed_generation")
    if _strict_int(boundary["tick"], path="snapshot.tick") != 0:
        raise _fail("snapshot.tick must be zero")
    if boundary["phase"] != "generation_complete":
        raise _fail("snapshot.phase must be generation_complete")
    if boundary["controller_status_at_clone"] not in {
        "generation_complete_waiting_persistence",
        "generation_complete_ready_to_resolve_terminal",
    }:
        raise _fail("snapshot.controller_status_at_clone mismatch")
    expected_snapshot_id = f"{controller_id}:owner:{owner_epoch}:boundary:{ordinal}"
    if boundary["snapshot_id"] != expected_snapshot_id:
        raise _fail("snapshot.snapshot_id mismatch")
    snapshot_digest = _validated_snapshot_digest(boundary)
    _require_bool(
        boundary,
        "snapshot_digest_pending_adapter",
        False,
        path="snapshot",
    )
    if type(boundary["persistence_ack_required"]) is not bool:
        raise _fail("snapshot.persistence_ack_required must be a boolean")
    _required_string(run_id, path="snapshot.run_id")
    return boundary, snapshot_digest


def _prepare_exact_mailbox_snapshot(
    runner: Any,
    snapshot: Mapping[str, Any],
) -> tuple[_PreparedBoundary, dict[str, Any], str]:
    boundary = _mapping(snapshot, path="snapshot")
    _require_fields(
        boundary,
        (
            "schema_version",
            "run_id",
            "controller_id",
            "owner_epoch",
            "snapshot_id",
            "snapshot_sequence",
            "boundary_ordinal",
            "completed_generation",
            "snapshot_digest",
            "snapshot_digest_sha256",
            "snapshot_digest_pending_adapter",
            "persistence_ack_required",
        ),
        path="snapshot",
    )
    if boundary["schema_version"] != BOUNDARY_SCHEMA_VERSION:
        raise _fail("snapshot.schema_version mismatch")
    run_id = _required_string(boundary["run_id"], path="snapshot.run_id")
    controller_id = _required_string(
        boundary["controller_id"], path="snapshot.controller_id"
    )
    owner_epoch = _strict_int(
        boundary["owner_epoch"], path="snapshot.owner_epoch", minimum=1
    )
    sequence = _strict_int(
        boundary["snapshot_sequence"], path="snapshot.snapshot_sequence", minimum=1
    )
    ordinal = _strict_int(
        boundary["boundary_ordinal"], path="snapshot.boundary_ordinal", minimum=1
    )
    if ordinal != sequence:
        raise _fail("snapshot.boundary_ordinal does not match snapshot_sequence")
    completed_generation = _strict_int(
        boundary["completed_generation"], path="snapshot.completed_generation"
    )
    snapshot_digest = _lowercase_sha256(
        boundary["snapshot_digest_sha256"], path="snapshot.snapshot_digest_sha256"
    )
    if (
        _lowercase_sha256(
            boundary["snapshot_digest"], path="snapshot.snapshot_digest"
        )
        != snapshot_digest
    ):
        raise _fail("snapshot digest fields disagree")
    _require_bool(
        boundary,
        "snapshot_digest_pending_adapter",
        False,
        path="snapshot",
    )
    if type(boundary["persistence_ack_required"]) is not bool:
        raise _fail("snapshot.persistence_ack_required must be a boolean")

    sim_cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if sim_cfg is None or env_cfg is None:
        raise _fail("runner.sim_cfg and runner.env_cfg are required")
    if not bool(getattr(sim_cfg, "behavior_log_enable", True)):
        raise _fail("Architecture B exact reporting requires behavior_log_enable")
    if not bool(getattr(sim_cfg, "lineage_observability_enable", True)):
        raise _fail("Architecture B exact reporting requires lineage_observability_enable")

    validation_boundary = dict(boundary)
    reporting = _mapping(
        boundary.get("generation_end_reporting_frame"),
        path="snapshot.generation_end_reporting_frame",
    )
    post = _mapping(
        boundary.get("post_turnover_checkpoint_projection"),
        path="snapshot.post_turnover_checkpoint_projection",
    )
    validation_boundary.setdefault(
        "reporting_projection_digest_manifests",
        {"source_projection": checkpoint_source_projection_digest_manifest(reporting)},
    )
    validation_boundary.setdefault(
        "checkpoint_projection_digest_manifests",
        {"source_projection": checkpoint_source_projection_digest_manifest(post)},
    )
    validation_boundary.setdefault(
        "tensor_manifest_boundary_bytes",
        {
            "generation_end_reporting_frame": boundary.get(
                "generation_end_reporting_frame_tensor_bytes"
            ),
            "post_turnover_checkpoint_projection": boundary.get(
                "post_turnover_projection_tensor_bytes"
            ),
            "total": _strict_int(
                boundary.get("generation_end_reporting_frame_tensor_bytes"),
                path="snapshot.generation_end_reporting_frame_tensor_bytes",
            )
            + _strict_int(
                boundary.get("post_turnover_projection_tensor_bytes"),
                path="snapshot.post_turnover_projection_tensor_bytes",
            ),
        },
    )
    try:
        prepared, _cursor, _tensor_bytes = _prepare_boundary(
            runner,
            validation_boundary,
            index=sequence - 1,
            run_id=run_id,
            controller_id=controller_id,
            owner_epoch=owner_epoch,
            expected_generation=completed_generation,
            expected_snapshot_sequence=sequence,
            previous_event_cursor=None,
        )
    except ArchitectureBReportingError:
        raise
    except Exception as exc:
        raise _fail(f"snapshot boundary validation failed: {exc}") from exc
    return prepared, boundary, str(boundary["snapshot_digest_sha256"])


def _snapshot_identity(
    boundary: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    identity = {
        "run_id": str(boundary["run_id"]),
        "controller_id": str(boundary["controller_id"]),
        "owner_epoch": int(boundary["owner_epoch"]),
        "snapshot_id": str(boundary["snapshot_id"]),
        "snapshot_sequence": int(boundary["snapshot_sequence"]),
        "boundary_ordinal": int(boundary["boundary_ordinal"]),
        "completed_generation": int(boundary["completed_generation"]),
        "next_generation": int(boundary["next_generation"]),
    }
    digest = _sha256_bytes(
        _canonical_json_bytes(identity, path="snapshot identity")
    )
    return identity, digest


def _due_flag(
    due_mask: Mapping[str, Any],
    *names: str,
    default: bool = False,
) -> bool:
    values: list[bool] = []
    for name in names:
        if name not in due_mask:
            continue
        value = due_mask[name]
        if type(value) is not bool:
            raise _fail(f"snapshot.due_mask.{name} must be a boolean")
        values.append(value)
    if len(set(values)) > 1:
        raise _fail(f"snapshot.due_mask aliases disagree: {list(names)}")
    return values[0] if values else default


def _cancellation_checkpoint_observed(boundary: Mapping[str, Any]) -> bool:
    """Validate the native-only cancellation checkpoint override on a snapshot."""

    observed_present = "cancellation_observed_at_snapshot_cutoff" in boundary
    requested_present = "cancellation_checkpoint_requested" in boundary
    if observed_present != requested_present:
        raise _fail(
            "snapshot cancellation observation and checkpoint request must be "
            "declared together"
        )
    observed = boundary.get("cancellation_observed_at_snapshot_cutoff", False)
    if type(observed) is not bool:
        raise _fail(
            "snapshot.cancellation_observed_at_snapshot_cutoff must be a boolean"
        )
    requested = boundary.get("cancellation_checkpoint_requested", False)
    if type(requested) is not bool:
        raise _fail("snapshot.cancellation_checkpoint_requested must be a boolean")
    if requested is not observed:
        raise _fail(
            "snapshot cancellation checkpoint request contradicts its observation"
        )
    for field in ("success_marker_allowed", "promotion_result_allowed"):
        if observed or field in boundary:
            _require_bool(boundary, field, False, path="snapshot")
    return observed


def _optional_cursor_int(
    cursor: Mapping[str, Any],
    field: str,
    *,
    path: str,
) -> int | None:
    if field not in cursor:
        return None
    return _strict_int(cursor[field], path=f"{path}.{field}")


def _required_cursor_int(
    cursor: Mapping[str, Any],
    field: str,
    *,
    path: str,
) -> int:
    value = _optional_cursor_int(cursor, field, path=path)
    if value is None:
        raise _fail(f"{path}.{field} is required")
    return value


def _resolved_persistence_schedule(
    boundary: Mapping[str, Any],
    prepared: _PreparedBoundary | None,
) -> _ResolvedPersistenceSchedule:
    explicit = "due_mask" in boundary
    generation = _strict_int(
        boundary["completed_generation"], path="snapshot.completed_generation"
    )
    next_generation = _strict_int(
        boundary["next_generation"], path="snapshot.next_generation"
    )
    generation_range = [generation, next_generation]
    if not explicit:
        if prepared is None:
            raise _fail("legacy mailbox snapshots require the exact-reporting facet")
        lifecycle_range = [
            prepared.lifecycle_start_cursor,
            prepared.lifecycle_end_cursor,
        ]
        cursor_ranges = {
            "exact_reporting": generation_range,
            "behaviors": generation_range,
            "metrics": generation_range,
            "lineage": generation_range,
            "typed_lifecycle": lifecycle_range,
        }
        return _ResolvedPersistenceSchedule(
            explicit=False,
            due_compact_progress=False,
            due_exact_reporting=True,
            due_chronicle=False,
            due_checkpoint=False,
            schedule_state={
                "schema_version": "eiel_architecture_b_implicit_schedule_v1",
                "exact_reporting_every_complete_generations": 1,
            },
            terminal_candidate="continue",
            cursor_before={},
            cursor_after={},
            payload_byte_counts={},
            channel_cursor_ranges=cursor_ranges,
            acknowledgement_cursor_ranges=cursor_ranges,
            due_facets=("behaviors", "lineage", "metrics", "typed_lifecycle"),
        )

    _require_fields(
        boundary,
        (
            "schedule_state",
            "compact_progress",
            "payload_profile",
            "expected_acknowledged_channel_cursor_ranges",
        ),
        path="snapshot",
    )
    due_mask = _mapping(boundary["due_mask"], path="snapshot.due_mask")
    allowed_due_fields = {
        "progress",
        "compact_progress",
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
        "chronicle",
        "scientific_chronicle",
        "scientific_chronicle_when_enabled",
        "checkpoint",
        "checkpoint_when_due",
        "final",
    }
    unsupported = sorted(set(due_mask) - allowed_due_fields)
    if unsupported:
        raise _fail(f"snapshot.due_mask has unsupported fields: {unsupported}")
    due_progress = _due_flag(due_mask, "progress", "compact_progress")
    due_exact = _due_flag(due_mask, "exact_reporting")
    due_chronicle = _due_flag(
        due_mask,
        "chronicle",
        "scientific_chronicle",
        "scientific_chronicle_when_enabled",
    )
    due_checkpoint = _due_flag(due_mask, "checkpoint", "checkpoint_when_due")
    cancellation_checkpoint_observed = _cancellation_checkpoint_observed(boundary)
    for facet in ("behaviors", "metrics", "lineage", "typed_lifecycle"):
        if facet in due_mask and _due_flag(due_mask, facet) != due_exact:
            raise _fail(
                f"snapshot.due_mask.{facet} must be co-scheduled with exact_reporting"
            )
    if due_chronicle and not due_exact:
        raise _fail("snapshot Chronicle output requires exact_reporting at the boundary")
    if not any((due_progress, due_exact, due_checkpoint)):
        raise _fail("snapshot.due_mask must schedule at least one observer")
    due_final = _due_flag(due_mask, "final")
    if due_final and not (due_exact or due_checkpoint):
        raise _fail("a final snapshot must schedule exact_reporting or checkpoint")
    if cancellation_checkpoint_observed and not due_checkpoint:
        raise _fail(
            "cancellation-observed snapshot must force a checkpoint-capable boundary"
        )
    acknowledgement_required = due_exact or due_chronicle or due_checkpoint
    if bool(boundary["persistence_ack_required"]) != acknowledgement_required:
        raise _fail(
            "snapshot.persistence_ack_required contradicts the required due facets"
        )

    schedule_state = _mapping(
        boundary["schedule_state"], path="snapshot.schedule_state"
    )
    schedule_fields = {
        "schema_version",
        "schedule_origin_generation",
        "imported_boundary_ordinal",
        "progress_every_generations",
        "report_every_generations",
        "checkpoint_every_generations",
        "explicit_next_generations",
    }
    optional_schedule_fields = {"checkpoint_on_cancel"}
    if not schedule_fields.issubset(schedule_state) or (
        set(schedule_state) - schedule_fields - optional_schedule_fields
    ):
        raise _fail("snapshot.schedule_state fields do not match the native schedule schema")
    checkpoint_on_cancel: bool | None = None
    if "checkpoint_on_cancel" in schedule_state:
        checkpoint_on_cancel = schedule_state["checkpoint_on_cancel"]
        if type(checkpoint_on_cancel) is not bool:
            raise _fail(
                "snapshot.schedule_state.checkpoint_on_cancel must be a boolean"
            )
    if cancellation_checkpoint_observed and checkpoint_on_cancel is False:
        raise _fail(
            "cancellation-observed snapshot contradicts disabled checkpoint_on_cancel"
        )
    if schedule_state["schema_version"] != "eiel_native_snapshot_schedule_v1":
        raise _fail("snapshot.schedule_state.schema_version mismatch")
    schedule_origin = _strict_int(
        schedule_state["schedule_origin_generation"],
        path="snapshot.schedule_state.schedule_origin_generation",
    )
    imported_boundary_ordinal = _strict_int(
        schedule_state["imported_boundary_ordinal"],
        path="snapshot.schedule_state.imported_boundary_ordinal",
    )
    progress_cadence = _strict_int(
        schedule_state["progress_every_generations"],
        path="snapshot.schedule_state.progress_every_generations",
    )
    report_cadence = _strict_int(
        schedule_state["report_every_generations"],
        path="snapshot.schedule_state.report_every_generations",
    )
    checkpoint_cadence = _strict_int(
        schedule_state["checkpoint_every_generations"],
        path="snapshot.schedule_state.checkpoint_every_generations",
    )
    raw_explicit_generations = schedule_state["explicit_next_generations"]
    if not isinstance(raw_explicit_generations, list):
        raise _fail(
            "snapshot.schedule_state.explicit_next_generations must be a list"
        )
    explicit_generations = [
        _strict_int(
            value,
            path=(
                "snapshot.schedule_state.explicit_next_generations"
                f"[{index}]"
            ),
        )
        for index, value in enumerate(raw_explicit_generations)
    ]
    if explicit_generations != sorted(set(explicit_generations)):
        raise _fail(
            "snapshot.schedule_state.explicit_next_generations must be sorted "
            "and unique"
        )
    if any(generation <= schedule_origin for generation in explicit_generations):
        raise _fail(
            "snapshot.schedule_state.explicit_next_generations must contain "
            "only future coordinates after schedule_origin_generation"
        )
    ordinal = int(boundary["boundary_ordinal"])
    if imported_boundary_ordinal >= ordinal:
        raise _fail(
            "snapshot.schedule_state.imported_boundary_ordinal must precede "
            "the current boundary"
        )
    schedule_coordinate = next_generation - schedule_origin
    if schedule_coordinate <= 0:
        raise _fail("snapshot schedule coordinate must be positive")
    if due_progress != (
        due_final
        or (progress_cadence > 0 and schedule_coordinate % progress_cadence == 0)
    ):
        raise _fail("snapshot progress due mask contradicts schedule_state")
    scheduled_exact_due = (
        report_cadence > 0 and schedule_coordinate % report_cadence == 0
    )
    if (
        (scheduled_exact_due and not due_exact)
        or (due_exact and not (scheduled_exact_due or due_final))
    ):
        raise _fail("snapshot exact-reporting due mask contradicts schedule_state")
    scheduled_checkpoint_due = (
        (
            checkpoint_cadence > 0
            and schedule_coordinate % checkpoint_cadence == 0
        )
        or next_generation in explicit_generations
    )
    if due_checkpoint != (
        scheduled_checkpoint_due or cancellation_checkpoint_observed
    ):
        raise _fail("snapshot checkpoint due mask contradicts schedule_state")

    compact_progress = _mapping(
        boundary["compact_progress"], path="snapshot.compact_progress"
    )
    _require_fields(
        compact_progress,
        ("completed_generation", "next_generation", "tick", "alive_count"),
        path="snapshot.compact_progress",
    )
    if (
        _strict_int(
            compact_progress["completed_generation"],
            path="snapshot.compact_progress.completed_generation",
        )
        != generation
        or _strict_int(
            compact_progress["next_generation"],
            path="snapshot.compact_progress.next_generation",
        )
        != next_generation
        or _strict_int(
            compact_progress["tick"], path="snapshot.compact_progress.tick"
        )
        != 0
    ):
        raise _fail("snapshot.compact_progress boundary coordinates are inconsistent")
    _strict_int(
        compact_progress["alive_count"],
        path="snapshot.compact_progress.alive_count",
    )
    expected_profile = (
        "canonical_checkpoint"
        if due_checkpoint
        else "exact_report" if due_exact else "progress_only"
    )
    if boundary["payload_profile"] != expected_profile:
        raise _fail("snapshot.payload_profile contradicts the due mask")

    expected_ranges_raw = _mapping(
        boundary["expected_acknowledged_channel_cursor_ranges"],
        path="snapshot.expected_acknowledged_channel_cursor_ranges",
    )
    expected_ranges: dict[str, list[int]] = {}
    for channel, raw_range in expected_ranges_raw.items():
        values = _sequence(
            raw_range,
            path=(
                "snapshot.expected_acknowledged_channel_cursor_ranges."
                f"{channel}"
            ),
        )
        if len(values) != 2:
            raise _fail(f"snapshot acknowledgement range {channel} must have two values")
        start = _strict_int(
            values[0],
            path=f"snapshot.expected_acknowledged_channel_cursor_ranges.{channel}[0]",
        )
        end = _strict_int(
            values[1],
            path=f"snapshot.expected_acknowledged_channel_cursor_ranges.{channel}[1]",
        )
        if end < start:
            raise _fail(f"snapshot acknowledgement range {channel} is reversed")
        expected_ranges[channel] = [start, end]

    required_exact_ranges = {
        "exact_reporting": generation_range,
        "behaviors": generation_range,
        "metrics": generation_range,
        "lineage": generation_range,
    }
    if due_exact:
        if prepared is None:
            raise _fail("exact-reporting snapshot is missing its required payload facet")
        typed_lifecycle_range = expected_ranges.get("typed_lifecycle")
        if (
            typed_lifecycle_range is None
            or typed_lifecycle_range[1] != prepared.lifecycle_end_cursor
            or typed_lifecycle_range[0] > prepared.lifecycle_start_cursor
        ):
            raise _fail(
                "native mailbox shared schema mismatch: typed_lifecycle range "
                "must cover every event since the prior acknowledged cursor"
            )
        required_exact_ranges["typed_lifecycle"] = typed_lifecycle_range
        for channel, expected in required_exact_ranges.items():
            if expected_ranges.get(channel) != expected:
                raise _fail(
                    "native mailbox shared schema mismatch: expected exact "
                    f"acknowledgement range {channel}={expected}"
                )
    elif any(channel in expected_ranges for channel in required_exact_ranges):
        raise _fail("progress/checkpoint-only snapshot acknowledges non-due reporting rows")
    if due_chronicle:
        if "chronicle" not in expected_ranges:
            raise _fail("snapshot Chronicle acknowledgement range is required")
    if due_checkpoint:
        expected_checkpoint_range = [next_generation, next_generation + 1]
        if expected_ranges.get("checkpoint_generation") != expected_checkpoint_range:
            raise _fail(
                "native mailbox shared schema mismatch: expected checkpoint "
                f"acknowledgement range checkpoint_generation={expected_checkpoint_range}"
            )
    elif "checkpoint_generation" in expected_ranges:
        raise _fail("snapshot acknowledges a non-due checkpoint")
    required_range_names = set(required_exact_ranges if due_exact else ())
    if due_chronicle:
        required_range_names.add("chronicle")
    if due_checkpoint:
        required_range_names.add("checkpoint_generation")
    if set(expected_ranges) != required_range_names:
        raise _fail(
            "snapshot acknowledgement cursor range fields do not exactly match due channels"
        )

    due_facets: list[str] = []
    if due_progress:
        due_facets.append("compact_progress")
    if due_exact:
        due_facets.extend(("behaviors", "lineage", "metrics", "typed_lifecycle"))
    if due_chronicle:
        due_facets.append("chronicle")
    if due_checkpoint:
        due_facets.append("checkpoint")
    return _ResolvedPersistenceSchedule(
        explicit=True,
        due_compact_progress=due_progress,
        due_exact_reporting=due_exact,
        due_chronicle=due_chronicle,
        due_checkpoint=due_checkpoint,
        schedule_state=schedule_state,
        terminal_candidate=(
            "cancelled_pending_durable_acknowledgement"
            if cancellation_checkpoint_observed
            else "run_complete" if due_final else "continue"
        ),
        cursor_before={},
        cursor_after={},
        payload_byte_counts={
            "generation_end_reporting_frame": _strict_int(
                boundary["generation_end_reporting_frame_tensor_bytes"],
                path="snapshot.generation_end_reporting_frame_tensor_bytes",
            ),
            "post_turnover_projection": _strict_int(
                boundary["post_turnover_projection_tensor_bytes"],
                path="snapshot.post_turnover_projection_tensor_bytes",
            ),
        },
        channel_cursor_ranges=expected_ranges,
        acknowledgement_cursor_ranges=expected_ranges,
        due_facets=tuple(due_facets),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _snapshot_identity_lock(root: Path, identity_digest: str):
    import fcntl

    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{identity_digest}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_existing_snapshot_directory(
    identity_dir: Path,
    expected_files: Mapping[str, bytes],
) -> bool:
    if not identity_dir.exists():
        return False
    if not identity_dir.is_dir():
        raise _fail(f"snapshot identity path is not a directory: {identity_dir}")
    actual_names = {
        str(path.relative_to(identity_dir))
        for path in identity_dir.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_names - set(expected_files))
    if unexpected:
        raise _fail(
            "snapshot identity directory contains unexpected persisted artifacts: "
            f"{unexpected}"
        )
    for name in sorted(actual_names):
        if (identity_dir / name).read_bytes() != expected_files[name]:
            raise _fail(
                "same snapshot identity has conflicting persisted content: "
                f"{name}"
            )
    complete = SNAPSHOT_COMPLETION_MARKER_NAME in actual_names
    if complete and actual_names != set(expected_files):
        missing = sorted(set(expected_files) - actual_names)
        raise _fail(
            "completed snapshot identity directory is missing artifacts: "
            f"{missing}"
        )
    return complete


def _artifact_record(
    *,
    path: str,
    facet: str,
    payload: bytes,
    cursor_range: Sequence[int],
) -> dict[str, Any]:
    return {
        "path": path,
        "facet": facet,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "cursor_range": [int(value) for value in cursor_range],
    }


def _checkpoint_artifact_records(
    transaction_dir: Path,
    checkpoint_dir: Path,
    *,
    cursor_range: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        records.append(
            _artifact_record(
                path=str(path.relative_to(transaction_dir)),
                facet="checkpoint",
                payload=payload,
                cursor_range=cursor_range,
            )
        )
    if not records:
        raise _fail("due checkpoint serializer produced no artifacts")
    return records


def _persist_due_checkpoint(
    transaction_dir: Path,
    boundary: Mapping[str, Any],
    identity: Mapping[str, Any],
    schedule: _ResolvedPersistenceSchedule,
    *,
    metrics_logger_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_save_projection
    from native_runtime.checkpoint_digests import (
        checkpoint_source_projection_authority_manifest,
    )

    projection = _mapping(
        boundary.get("checkpoint_projection"),
        path="snapshot.checkpoint_projection",
    )
    expected_coordinate = (
        f"generation:{identity['next_generation']}:tick:0"
    )
    if boundary.get("checkpoint_coordinate") != expected_coordinate:
        raise _fail("snapshot.checkpoint_coordinate is inconsistent")
    source_digest = str(
        checkpoint_source_projection_authority_manifest(projection)[
            "source_projection_sha256"
        ]
    )
    lifecycle_events = typed_lifecycle_events_from_projection(projection)
    lifecycle_cursor = validated_typed_lifecycle_event_cursor(
        projection, lifecycle_events
    )
    architecture_b_identity = {
        "run_id": identity["run_id"],
        "controller_id": identity["controller_id"],
        "owner_epoch": identity["owner_epoch"],
        "snapshot_id": identity["snapshot_id"],
        "snapshot_sequence": identity["snapshot_sequence"],
        "snapshot_digest_sha256": boundary["snapshot_digest_sha256"],
        "next_generation": identity["next_generation"],
        "tick": 0,
        "phase": "generation_boundary",
        "source_projection_sha256": source_digest,
    }
    observer_continuation = _mapping(
        boundary.get("observer_continuation"),
        path="snapshot.observer_continuation",
    )
    expected_continuation_scalars = {
        "checkpoint_next_generation": int(identity["next_generation"]),
        "tick": 0,
        "typed_lifecycle_event_cursor": lifecycle_cursor,
        "exact_reporting_boundary_ordinal": int(identity["snapshot_sequence"]),
        "chronicle_observation_ordinal": int(identity["snapshot_sequence"]),
        "progress_sequence": int(identity["snapshot_sequence"]),
    }
    if observer_continuation.get("schema_version") != (
        "eiel_architecture_b_observer_continuation_v1"
    ):
        raise _fail("snapshot.observer_continuation.schema_version mismatch")
    for field, expected_value in expected_continuation_scalars.items():
        if _strict_int(
            observer_continuation.get(field),
            path=f"snapshot.observer_continuation.{field}",
        ) != expected_value:
            raise _fail(
                f"snapshot.observer_continuation.{field} contradicts the snapshot"
            )
    continuation_ranges = _mapping(
        observer_continuation.get("acknowledged_channel_cursor_ranges"),
        path=(
            "snapshot.observer_continuation."
            "acknowledged_channel_cursor_ranges"
        ),
    )
    checkpoint_range = _sequence_range(
        continuation_ranges.get("checkpoint_generation"),
        path=(
            "snapshot.observer_continuation."
            "acknowledged_channel_cursor_ranges.checkpoint_generation"
        ),
    )
    if checkpoint_range != schedule.channel_cursor_ranges["checkpoint_generation"]:
        raise _fail(
            "snapshot checkpoint continuation range contradicts the due-channel "
            "cursor candidate"
        )
    if schedule.due_exact_reporting:
        for channel in (
            "exact_reporting",
            "behaviors",
            "metrics",
            "lineage",
            "typed_lifecycle",
        ):
            continuation_range = _sequence_range(
                continuation_ranges.get(channel),
                path=(
                    "snapshot.observer_continuation."
                    f"acknowledged_channel_cursor_ranges.{channel}"
                ),
            )
            if continuation_range != schedule.channel_cursor_ranges[channel]:
                raise _fail(
                    "snapshot observer continuation contradicts the due-channel "
                    f"cursor candidate for {channel}"
                )
    else:
        unreported_generation_range = [
            int(identity["completed_generation"]),
            int(identity["next_generation"]),
        ]
        for channel in ("exact_reporting", "behaviors", "metrics", "lineage"):
            continuation_range = _sequence_range(
                continuation_ranges.get(channel),
                path=(
                    "snapshot.observer_continuation."
                    f"acknowledged_channel_cursor_ranges.{channel}"
                ),
            )
            if continuation_range != unreported_generation_range:
                raise _fail(
                    "checkpoint-only observer continuation does not preserve the "
                    f"unreported exact channel {channel}"
                )
        lifecycle_range = _sequence_range(
            continuation_ranges.get("typed_lifecycle"),
            path=(
                "snapshot.observer_continuation."
                "acknowledged_channel_cursor_ranges.typed_lifecycle"
            ),
        )
        if lifecycle_range[1] != lifecycle_cursor:
            raise _fail(
                "checkpoint-only observer continuation does not preserve the "
                "unreported typed lifecycle suffix"
            )
        deduplication = _mapping(
            observer_continuation.get("deduplication"),
            path="snapshot.observer_continuation.deduplication",
        )
        record_keys = _sequence(
            deduplication.get("acknowledged_record_keys"),
            path=(
                "snapshot.observer_continuation.deduplication."
                "acknowledged_record_keys"
            ),
        )
        current_exact_key = (
            f"generation:{identity['completed_generation']}:exact_reporting"
        )
        if current_exact_key in record_keys:
            raise _fail(
                "checkpoint-only observer continuation falsely acknowledges "
                "unreported exact output"
            )
    checkpoint_dir = transaction_dir / "checkpoint"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(checkpoint_dir),
        generation_index=int(identity["next_generation"]),
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=architecture_b_identity,
        observer_continuation=observer_continuation,
        metrics_logger_state=metrics_logger_state,
    )
    return _checkpoint_artifact_records(
        transaction_dir,
        checkpoint_dir,
        cursor_range=schedule.channel_cursor_ranges["checkpoint_generation"],
    )


def _validate_published_snapshot(
    identity_dir: Path,
    *,
    expected: Mapping[str, Any],
) -> str | None:
    if not identity_dir.exists():
        return None
    if not identity_dir.is_dir():
        raise _fail(f"snapshot identity path is not a directory: {identity_dir}")
    manifest_path = identity_dir / SNAPSHOT_ARTIFACT_MANIFEST_NAME
    completion_path = identity_dir / SNAPSHOT_COMPLETION_MARKER_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise _fail("snapshot identity directory is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    completion_bytes = completion_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
        completion = json.loads(completion_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("persisted snapshot metadata is not valid JSON") from exc
    if manifest_bytes != _canonical_json_bytes(
        manifest, path="persisted snapshot artifact manifest"
    ):
        raise _fail("persisted snapshot artifact manifest is not canonical")
    if completion_bytes != _canonical_json_bytes(
        completion, path="persisted snapshot completion marker"
    ):
        raise _fail("persisted snapshot completion marker is not canonical")
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise _fail(
                "same snapshot identity has conflicting persisted content: "
                f"{field}"
            )
    manifest_digest = _sha256_bytes(manifest_bytes)
    if (
        completion.get("schema_version") != SNAPSHOT_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != "complete"
        or completion.get("publication_scope") != "snapshot_persistence_only"
        or completion.get("terminal_candidate")
        != manifest.get("terminal_candidate")
        or completion.get("cancellation_checkpoint_observed")
        != manifest.get("cancellation_checkpoint_observed")
        or completion.get("success_marker_allowed") is not False
        or completion.get("promotion_result_allowed") is not False
        or completion.get("artifact_manifest_path")
        != SNAPSHOT_ARTIFACT_MANIFEST_NAME
        or completion.get("artifact_manifest_digest_sha256") != manifest_digest
    ):
        raise _fail("persisted snapshot completion marker is invalid")
    expected_files = {
        SNAPSHOT_ARTIFACT_MANIFEST_NAME,
        SNAPSHOT_COMPLETION_MARKER_NAME,
    }
    artifacts = _sequence(
        manifest.get("artifacts"), path="persisted snapshot manifest.artifacts"
    )
    channel_ranges = _mapping(
        manifest.get("channel_cursor_ranges"),
        path="persisted snapshot manifest.channel_cursor_ranges",
    )
    artifact_paths: set[str] = set()
    artifact_bytes = 0
    for index, item in enumerate(artifacts):
        record = _mapping(item, path=f"persisted snapshot manifest.artifacts[{index}]")
        relative = _required_string(
            record.get("path"), path=f"persisted snapshot manifest.artifacts[{index}].path"
        )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise _fail("persisted snapshot artifact path escapes its identity directory")
        if relative in artifact_paths:
            raise _fail(f"persisted snapshot artifact path is duplicated: {relative}")
        artifact_paths.add(relative)
        path = identity_dir / relative_path
        if not path.is_file():
            raise _fail(f"persisted snapshot artifact is missing: {relative}")
        payload = path.read_bytes()
        facet = _required_string(
            record.get("facet"),
            path=f"persisted snapshot manifest.artifacts[{index}].facet",
        )
        record_range = _sequence_range(
            record.get("cursor_range"),
            path=f"persisted snapshot manifest.artifacts[{index}].cursor_range",
        )
        channel = "checkpoint_generation" if facet == "checkpoint" else facet
        if channel in channel_ranges:
            declared_range = _sequence_range(
                channel_ranges[channel],
                path=f"persisted snapshot manifest.channel_cursor_ranges.{channel}",
            )
            if record_range != declared_range:
                raise _fail(
                    f"persisted snapshot artifact cursor range mismatch: {relative}"
                )
        if (
            _strict_int(
                record.get("bytes"),
                path=f"persisted snapshot manifest.artifacts[{index}].bytes",
            )
            != len(payload)
            or _lowercase_sha256(
                record.get("sha256"),
                path=f"persisted snapshot manifest.artifacts[{index}].sha256",
            )
            != _sha256_bytes(payload)
        ):
            raise _fail(f"persisted snapshot artifact digest mismatch: {relative}")
        artifact_bytes += len(payload)
        expected_files.add(relative)
    if (
        _strict_int(
            manifest.get("artifact_count"),
            path="persisted snapshot manifest.artifact_count",
        )
        != len(artifacts)
        or _strict_int(
            manifest.get("artifact_bytes"),
            path="persisted snapshot manifest.artifact_bytes",
        )
        != artifact_bytes
    ):
        raise _fail("persisted snapshot artifact totals do not match its manifest")
    actual_files = {
        str(path.relative_to(identity_dir))
        for path in identity_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise _fail("persisted snapshot artifact inventory does not match its manifest")
    return manifest_digest


def _published_manifest(identity_dir: Path) -> dict[str, Any]:
    if _validate_published_snapshot(identity_dir, expected={}) is None:
        raise _fail(f"persisted snapshot disappeared: {identity_dir}")
    try:
        return _mapping(
            json.loads(
                (identity_dir / SNAPSHOT_ARTIFACT_MANIFEST_NAME).read_bytes()
            ),
            path=f"persisted snapshot manifest {identity_dir.name}",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"persisted snapshot manifest is invalid: {identity_dir}") from exc


def _ranges_overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    return int(left[0]) < int(right[1]) and int(right[0]) < int(left[1])


def _sequence_range(value: Any, *, path: str) -> list[int]:
    values = _sequence(value, path=path)
    if len(values) != 2:
        raise _fail(f"{path} must contain exactly two cursor values")
    start = _strict_int(values[0], path=f"{path}[0]")
    end = _strict_int(values[1], path=f"{path}[1]")
    if end < start:
        raise _fail(f"{path} is reversed")
    return [start, end]


def _validate_prior_channel_cursor_ranges(
    root: Path,
    identity_dir: Path,
    identity: Mapping[str, Any],
    schedule: _ResolvedPersistenceSchedule,
) -> None:
    """Reject duplicate or discontinuous exact rows already published for the run."""

    current_sequence = int(identity["snapshot_sequence"])
    current_ranges = schedule.channel_cursor_ranges
    prior: list[tuple[int, dict[str, Any]]] = []
    for candidate in sorted(root.glob("snapshot-*")):
        if candidate == identity_dir or not candidate.is_dir():
            continue
        manifest = _published_manifest(candidate)
        prior_identity = _mapping(
            manifest.get("snapshot_identity"),
            path=f"persisted snapshot manifest {candidate.name}.snapshot_identity",
        )
        if prior_identity.get("run_id") != identity["run_id"]:
            continue
        sequence = _strict_int(
            prior_identity.get("snapshot_sequence"),
            path=(
                f"persisted snapshot manifest {candidate.name}."
                "snapshot_identity.snapshot_sequence"
            ),
            minimum=1,
        )
        if sequence >= current_sequence:
            raise _fail(
                "snapshot delivery order conflicts with an already published "
                f"run snapshot: existing_sequence={sequence} "
                f"current_sequence={current_sequence}"
            )
        prior.append((sequence, manifest))

    exact_channels = (
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
    )
    for _sequence, manifest in prior:
        persisted_ranges = _mapping(
            manifest.get("channel_cursor_ranges", {}),
            path="persisted snapshot manifest.channel_cursor_ranges",
        )
        for channel in exact_channels:
            if channel not in current_ranges or channel not in persisted_ranges:
                continue
            persisted = _sequence_range(
                persisted_ranges[channel],
                path=f"persisted snapshot channel range {channel}",
            )
            current = current_ranges[channel]
            if _ranges_overlap(persisted, current):
                raise _fail(
                    "snapshot channel cursor range duplicates already persisted rows: "
                    f"{channel}={current}"
                )
        if (
            "checkpoint_generation" in current_ranges
            and "checkpoint_generation" in persisted_ranges
        ):
            persisted_checkpoint = _sequence_range(
                persisted_ranges["checkpoint_generation"],
                path="persisted snapshot checkpoint_generation range",
            )
            if _ranges_overlap(
                persisted_checkpoint, current_ranges["checkpoint_generation"]
            ):
                raise _fail(
                    "snapshot checkpoint cursor range duplicates an already "
                    "published checkpoint"
                )

    if "typed_lifecycle" in current_ranges:
        typed_history: list[tuple[int, list[int]]] = []
        for sequence, manifest in prior:
            persisted_ranges = _mapping(
                manifest.get("channel_cursor_ranges", {}),
                path="persisted snapshot manifest.channel_cursor_ranges",
            )
            if "typed_lifecycle" in persisted_ranges:
                typed_history.append(
                    (
                        sequence,
                        _sequence_range(
                            persisted_ranges["typed_lifecycle"],
                            path="persisted snapshot typed_lifecycle range",
                        ),
                    )
                )
        if typed_history:
            previous_end = max(typed_history, key=lambda item: item[0])[1][1]
            current_start = int(current_ranges["typed_lifecycle"][0])
            if current_start != previous_end:
                raise _fail(
                    "snapshot typed_lifecycle cursor does not continue the last "
                    f"published exact boundary: expected={previous_end} "
                    f"actual={current_start}"
                )


def _failure_acknowledgement(
    *,
    request_id: str,
    identity: Mapping[str, Any],
    snapshot_digest: str,
    failure_code: str,
    failure_digest: str,
    acknowledged_ranges: Mapping[str, Sequence[int]],
) -> ArchitectureBSnapshotPersistenceAcknowledgement:
    return {
        "api_version": 1,
        "request_id": request_id,
        "expected_controller_id": str(identity["controller_id"]),
        "expected_owner_epoch": int(identity["owner_epoch"]),
        "snapshot_id": str(identity["snapshot_id"]),
        "snapshot_digest_sha256": snapshot_digest,
        "outcome": "failed",
        "artifact_manifest_digest_sha256": failure_digest,
        "acknowledged_channel_cursor_ranges": {
            str(channel): [int(value) for value in cursor_range]
            for channel, cursor_range in acknowledged_ranges.items()
        },
        "failure_code": failure_code,
    }


def _remove_success_markers(root: Path) -> list[str]:
    cleanup_errors: list[str] = []
    for name in (SNAPSHOT_COMPLETION_MARKER_NAME, "bundle_complete.json"):
        for path in root.rglob(name):
            try:
                path.unlink()
            except Exception as exc:
                cleanup_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return cleanup_errors


def _quarantine_persistence_failure(
    *,
    root: Path,
    identity_dir: Path,
    transaction_dir: Path,
    identity: Mapping[str, Any],
    identity_digest: str,
    snapshot_digest: str,
    snapshot_content_digest: str,
    failure_code: str,
    phase: str,
    operation: str,
    cause: BaseException,
) -> tuple[str, str]:
    quarantine_dir = root / ".quarantine" / f"snapshot-{identity_digest}"
    existing_failure = quarantine_dir / SNAPSHOT_FAILURE_RECORD_NAME
    if existing_failure.is_file():
        payload = existing_failure.read_bytes()
        record = json.loads(payload)
        return str(record["error_code"]), _sha256_bytes(payload)

    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=False, exist_ok=False)
    cleanup_errors: list[str] = []
    for destination_name, source in (
        ("incomplete_transaction", transaction_dir),
        ("incomplete_artifacts", identity_dir),
    ):
        if not source.exists():
            continue
        try:
            os.replace(source, quarantine_dir / destination_name)
        except Exception as exc:
            cleanup_errors.append(
                f"quarantine_move_{destination_name}: {type(exc).__name__}: {exc}"
            )
    cleanup_errors.extend(_remove_success_markers(quarantine_dir))
    message = f"{type(cause).__name__}: {cause}"
    failure_record = {
        "schema_version": SNAPSHOT_FAILURE_SCHEMA_VERSION,
        "error_sequence": 1,
        "error_code": failure_code,
        "error_class": type(cause).__name__,
        "origin": "python_snapshot_persistence_adapter",
        "phase": phase,
        "operation": operation,
        "run_id": identity["run_id"],
        "owner_epoch": identity["owner_epoch"],
        "generation": identity["completed_generation"],
        "tick": 0,
        "trajectory_transaction_state": "complete_generation_committed",
        "owner_integrity": "healthy_but_requires_failure_seal",
        "trajectory_commit_frontier": identity["completed_generation"],
        "snapshot_frontier": identity["completed_generation"],
        "output_ack_frontier": identity["completed_generation"] - 1,
        "durable_resume_frontier": identity["completed_generation"] - 1,
        "message": message[:2048],
        "retry_disposition": "prohibited_same_run_owner",
        "terminal_status": "failed",
        "success_marker_allowed": False,
        "promotion_result_allowed": False,
        "snapshot_id": identity["snapshot_id"],
        "snapshot_digest_sha256": snapshot_digest,
        "snapshot_content_digest_sha256": snapshot_content_digest,
        "cleanup_errors": cleanup_errors,
    }
    payload = _canonical_json_bytes(
        failure_record, path="snapshot persistence failure record"
    )
    _atomic_write_bytes(existing_failure, payload)
    _fsync_directory(quarantine_dir)
    return failure_code, _sha256_bytes(payload)


def persist_architecture_b_snapshot(
    runner: Any,
    snapshot: Mapping[str, Any],
    *,
    staging_root: str | Path | None = None,
    request_id: str | None = None,
) -> ArchitectureBSnapshotPersistenceAcknowledgement:
    """Persist one immutable mailbox snapshot without consulting the live owner.

    The staged facet envelopes are intentionally per-snapshot inputs. This
    adapter does not append or aggregate the final run-level public files.
    """

    boundary, snapshot_content_digest = _validated_mailbox_snapshot_header(snapshot)
    explicit_due = _mapping(boundary["due_mask"], path="snapshot.due_mask") if (
        "due_mask" in boundary
    ) else None
    due_exact = (
        True
        if explicit_due is None
        else _due_flag(explicit_due, "exact_reporting")
    )
    prepared: _PreparedBoundary | None = None
    if due_exact:
        prepared, _exact_boundary, _exact_content_digest = (
            _prepare_exact_mailbox_snapshot(runner, boundary)
        )
    schedule = _resolved_persistence_schedule(boundary, prepared)
    identity, identity_digest = _snapshot_identity(boundary)
    resolved_request_id = (
        f"persist-{identity['snapshot_sequence']}-{identity_digest[:16]}"
        if request_id is None
        else _required_string(request_id, path="request_id")
    )
    snapshot_digest = str(boundary["snapshot_digest_sha256"])
    expected_manifest = {
        "schema_version": SNAPSHOT_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "snapshot_identity_sha256": identity_digest,
        "snapshot_digest_sha256": snapshot_digest,
        "snapshot_content_digest_sha256": snapshot_content_digest,
        "due_facets": list(schedule.due_facets),
        "schedule_state": schedule.schedule_state,
        "terminal_candidate": schedule.terminal_candidate,
        "acknowledged_channel_cursor_ranges": (
            schedule.acknowledgement_cursor_ranges
        ),
        "channel_cursor_ranges": schedule.channel_cursor_ranges,
        "success_marker_allowed": False,
        "promotion_result_allowed": False,
        "cancellation_checkpoint_observed": (
            _cancellation_checkpoint_observed(boundary)
        ),
    }

    if staging_root is None:
        run_dir_value = str(getattr(runner, "run_dir", "") or "")
        if not run_dir_value:
            raise _fail("runner.run_dir is required when staging_root is omitted")
        root = Path(run_dir_value) / SNAPSHOT_STAGING_DIRECTORY
    else:
        root = Path(staging_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise _fail(f"snapshot persistence root creation failed: {exc}") from exc

    identity_dir = root / f"snapshot-{identity_digest}"
    transaction_dir = root / ".transactions" / f"snapshot-{identity_digest}"
    quarantine_dir = root / ".quarantine" / f"snapshot-{identity_digest}"
    accepted_metrics_state: dict[str, Any] | None = None
    with _snapshot_identity_lock(root, identity_digest):
        existing_failure = quarantine_dir / SNAPSHOT_FAILURE_RECORD_NAME
        if existing_failure.is_file():
            failure_bytes = existing_failure.read_bytes()
            failure = _mapping(
                json.loads(failure_bytes), path="snapshot persisted failure"
            )
            if (
                failure.get("snapshot_digest_sha256") != snapshot_digest
                or failure.get("snapshot_content_digest_sha256")
                != snapshot_content_digest
            ):
                raise _fail(
                    "same snapshot identity has conflicting persisted content: "
                    "durable failure record"
                )
            return _failure_acknowledgement(
                request_id=resolved_request_id,
                identity=identity,
                snapshot_digest=snapshot_digest,
                failure_code=str(failure["error_code"]),
                failure_digest=_sha256_bytes(failure_bytes),
                acknowledged_ranges=schedule.acknowledgement_cursor_ranges,
            )
        _validate_prior_channel_cursor_ranges(
            root,
            identity_dir,
            identity,
            schedule,
        )
        if identity_dir.exists() and not (
            identity_dir.is_dir()
            and (identity_dir / SNAPSHOT_ARTIFACT_MANIFEST_NAME).is_file()
            and (identity_dir / SNAPSHOT_COMPLETION_MARKER_NAME).is_file()
        ):
            interrupted = _fail(
                "a pre-existing published snapshot identity is incomplete"
            )
            code, digest = _quarantine_persistence_failure(
                root=root,
                identity_dir=identity_dir,
                transaction_dir=transaction_dir,
                identity=identity,
                identity_digest=identity_digest,
                snapshot_digest=snapshot_digest,
                snapshot_content_digest=snapshot_content_digest,
                failure_code="output_flush_failed",
                phase="snapshot_publication",
                operation="recover_incomplete_publication",
                cause=interrupted,
            )
            return _failure_acknowledgement(
                request_id=resolved_request_id,
                identity=identity,
                snapshot_digest=snapshot_digest,
                failure_code=code,
                failure_digest=digest,
                acknowledged_ranges=schedule.acknowledgement_cursor_ranges,
            )
        existing_manifest_digest = _validate_published_snapshot(
            identity_dir, expected=expected_manifest
        )
        if existing_manifest_digest is not None:
            return {
                "api_version": 1,
                "request_id": resolved_request_id,
                "expected_controller_id": str(identity["controller_id"]),
                "expected_owner_epoch": int(identity["owner_epoch"]),
                "snapshot_id": str(identity["snapshot_id"]),
                "snapshot_digest_sha256": snapshot_digest,
                "outcome": "success",
                "artifact_manifest_digest_sha256": existing_manifest_digest,
                "acknowledged_channel_cursor_ranges": (
                    schedule.acknowledgement_cursor_ranges
                ),
                "failure_code": None,
            }
        if transaction_dir.exists():
            interrupted = _fail("an incomplete snapshot transaction already exists")
            code, digest = _quarantine_persistence_failure(
                root=root,
                identity_dir=identity_dir,
                transaction_dir=transaction_dir,
                identity=identity,
                identity_digest=identity_digest,
                snapshot_digest=snapshot_digest,
                snapshot_content_digest=snapshot_content_digest,
                failure_code="output_flush_failed",
                phase="snapshot_publication",
                operation="recover_incomplete_transaction",
                cause=interrupted,
            )
            return _failure_acknowledgement(
                request_id=resolved_request_id,
                identity=identity,
                snapshot_digest=snapshot_digest,
                failure_code=code,
                failure_digest=digest,
                acknowledged_ranges=schedule.acknowledgement_cursor_ranges,
            )

        transaction_dir.parent.mkdir(parents=True, exist_ok=True)
        transaction_dir.mkdir(parents=False, exist_ok=False)
        operation = "artifact_derivation"
        failure_code = "exact_reporting_failed"
        try:
            common = {
                "source": "immutable_architecture_b_mailbox_snapshot",
                "snapshot_identity": identity,
                "snapshot_digest_sha256": snapshot_digest,
            }
            artifact_values: dict[str, Any] = {}
            checkpoint_metrics_state: dict[str, Any] | None = None
            if schedule.due_checkpoint:
                checkpoint_metrics_state = _current_metrics_continuation_state(runner)
            if schedule.due_exact_reporting:
                if prepared is None:
                    raise _fail("exact-reporting payload was not prepared")
                accepted_metrics_state = _advance_metrics_continuation_state(
                    runner,
                    prepared,
                    temporary_parent=transaction_dir,
                )
                if schedule.due_checkpoint:
                    checkpoint_metrics_state = accepted_metrics_state
            if schedule.due_compact_progress:
                artifact_values["compact_progress.json"] = {
                    "schema_version": "eiel_architecture_b_snapshot_progress_v1",
                    **common,
                    "progress": _mapping(
                        boundary["compact_progress"],
                        path="snapshot.compact_progress",
                    ),
                }
            if schedule.due_exact_reporting:
                if prepared is None:
                    raise _fail("exact-reporting payload was not prepared")
                generation_range = schedule.channel_cursor_ranges[
                    "exact_reporting"
                ]
                lifecycle_range = schedule.channel_cursor_ranges[
                    "typed_lifecycle"
                ]
                lineage_input = prepared.overlay["values"][
                    NATIVE_LINEAGE_METRICS_INPUT_KEY
                ]
                lifecycle_rows = typed_lifecycle_events_to_rows(
                    prepared.lifecycle_events,
                    start=lifecycle_range[0],
                    stop=lifecycle_range[1],
                )
                artifact_values.update(
                    {
                        "behaviors.json": {
                            "schema_version": (
                                "eiel_architecture_b_snapshot_behaviors_v1"
                            ),
                            **common,
                            "completed_generation_range": generation_range,
                            "row_count": len(prepared.behavior_rows),
                            "rows": list(prepared.behavior_rows),
                        },
                        "lineage.json": {
                            "schema_version": (
                                "eiel_architecture_b_snapshot_lineage_v1"
                            ),
                            **common,
                            "completed_generation_range": generation_range,
                            "record_count": int(lineage_input["record_count"]),
                            "native_lineage_metrics_input": lineage_input,
                        },
                        "metrics.json": {
                            "schema_version": (
                                "eiel_architecture_b_snapshot_metrics_v1"
                            ),
                            **common,
                            "completed_generation_range": generation_range,
                            "cohort_size": prepared.cohort_size,
                            "cohort_survivors": prepared.cohort_survivors,
                            "public_output_counts": prepared.counts,
                            "public_metrics_overlay": prepared.overlay,
                        },
                        "typed_lifecycle.json": {
                            "schema_version": (
                                "eiel_architecture_b_snapshot_typed_lifecycle_v1"
                            ),
                            **common,
                            "event_cursor_range": lifecycle_range,
                            "row_count": len(lifecycle_rows),
                            "cumulative_event_digest_sha256": (
                                typed_lifecycle_event_digest(
                                    prepared.lifecycle_events
                                )
                            ),
                            "rows": lifecycle_rows,
                        },
                    }
                )
            if schedule.due_chronicle:
                chronicle_inputs = boundary.get("chronicle_inputs")
                if chronicle_inputs is None:
                    raise _fail("due Chronicle snapshot is missing chronicle_inputs")
                artifact_values["chronicle.json"] = {
                    "schema_version": "eiel_architecture_b_snapshot_chronicle_v1",
                    **common,
                    "inputs": chronicle_inputs,
                }

            operation = "artifact_serialization"
            artifact_bytes = {
                name: _canonical_json_bytes(
                    value, path=f"snapshot artifact {name}"
                )
                for name, value in artifact_values.items()
            }
            operation = "artifact_write"
            artifact_records: list[dict[str, Any]] = []
            for name in sorted(artifact_bytes):
                facet = name.removesuffix(".json")
                payload = artifact_bytes[name]
                _atomic_write_bytes(transaction_dir / name, payload)
                cursor_range = schedule.channel_cursor_ranges.get(
                    facet,
                    [identity["completed_generation"], identity["next_generation"]],
                )
                artifact_records.append(
                    _artifact_record(
                        path=name,
                        facet=facet,
                        payload=payload,
                        cursor_range=cursor_range,
                    )
                )

            if schedule.due_checkpoint:
                if checkpoint_metrics_state is None:
                    raise _fail("checkpoint metrics continuation state is missing")
                operation = "checkpoint_serialization"
                failure_code = "checkpoint_persistence_failed"
                artifact_records.extend(
                    _persist_due_checkpoint(
                        transaction_dir,
                        boundary,
                        identity,
                        schedule,
                        metrics_logger_state=checkpoint_metrics_state,
                    )
                )
                failure_code = "output_flush_failed"

            projection_digests: dict[str, str] = {}
            if schedule.due_exact_reporting:
                projection_digests["generation_end_reporting_frame"] = str(
                    checkpoint_source_projection_digest_manifest(
                        _mapping(
                            boundary["generation_end_reporting_frame"],
                            path="snapshot.generation_end_reporting_frame",
                        )
                    )["digest"]
                )
                projection_digests["post_turnover_checkpoint_projection"] = str(
                    checkpoint_source_projection_digest_manifest(
                        _mapping(
                            boundary["post_turnover_checkpoint_projection"],
                            path="snapshot.post_turnover_checkpoint_projection",
                        )
                    )["digest"]
                )
            if schedule.due_checkpoint:
                projection_digests["checkpoint_projection"] = str(
                    checkpoint_source_projection_digest_manifest(
                        _mapping(
                            boundary["checkpoint_projection"],
                            path="snapshot.checkpoint_projection",
                        )
                    )["digest"]
                )

            operation = "manifest_serialization"
            manifest = {
                **expected_manifest,
                "source": "immutable_architecture_b_mailbox_snapshot",
                "snapshot_identity": identity,
                "source_projection_digests_sha256": projection_digests,
                "channel_cursor_ranges": schedule.channel_cursor_ranges,
                "payload_byte_counts": schedule.payload_byte_counts,
                "artifacts": artifact_records,
                "artifact_count": len(artifact_records),
                "artifact_bytes": sum(
                    int(record["bytes"]) for record in artifact_records
                ),
                "resident_owner_read": False,
                "owner_mutation": False,
                "trajectory_reentry": False,
                "final_run_files_aggregated": False,
            }
            manifest_bytes = _canonical_json_bytes(
                manifest, path="snapshot artifact manifest"
            )
            manifest_digest = _sha256_bytes(manifest_bytes)
            operation = "manifest_write"
            _atomic_write_bytes(
                transaction_dir / SNAPSHOT_ARTIFACT_MANIFEST_NAME,
                manifest_bytes,
            )
            operation = "artifact_flush"
            _fsync_directory(transaction_dir)
            completion = {
                "schema_version": SNAPSHOT_COMPLETION_SCHEMA_VERSION,
                "status": "complete",
                "publication_scope": "snapshot_persistence_only",
                "terminal_candidate": schedule.terminal_candidate,
                "cancellation_checkpoint_observed": (
                    _cancellation_checkpoint_observed(boundary)
                ),
                "success_marker_allowed": False,
                "promotion_result_allowed": False,
                "snapshot_identity_sha256": identity_digest,
                "snapshot_digest_sha256": snapshot_digest,
                "artifact_manifest_path": SNAPSHOT_ARTIFACT_MANIFEST_NAME,
                "artifact_manifest_digest_sha256": manifest_digest,
            }
            operation = "completion_marker_write"
            _atomic_write_bytes(
                transaction_dir / SNAPSHOT_COMPLETION_MARKER_NAME,
                _canonical_json_bytes(
                    completion, path="snapshot completion marker"
                ),
            )
            operation = "completion_flush"
            _fsync_directory(transaction_dir)
            operation = "snapshot_publish_rename"
            os.rename(transaction_dir, identity_dir)
            operation = "publication_flush"
            _fsync_directory(root)
            operation = "published_snapshot_verification"
            published_manifest_digest = _validate_published_snapshot(
                identity_dir, expected=expected_manifest
            )
            if published_manifest_digest is None:
                raise _fail("published snapshot identity directory disappeared")
            if published_manifest_digest != manifest_digest:
                raise _fail("published snapshot manifest digest changed after rename")
        except Exception as exc:
            if operation == "checkpoint_serialization":
                failure_code = "checkpoint_persistence_failed"
            elif operation in {
                "manifest_write",
                "artifact_flush",
                "completion_marker_write",
                "completion_flush",
                "snapshot_publish_rename",
                "publication_flush",
                "published_snapshot_verification",
            }:
                failure_code = "output_flush_failed"
            elif not (schedule.due_exact_reporting or schedule.due_chronicle):
                failure_code = "output_flush_failed"
            code, digest = _quarantine_persistence_failure(
                root=root,
                identity_dir=identity_dir,
                transaction_dir=transaction_dir,
                identity=identity,
                identity_digest=identity_digest,
                snapshot_digest=snapshot_digest,
                snapshot_content_digest=snapshot_content_digest,
                failure_code=failure_code,
                phase="snapshot_persistence",
                operation=operation,
                cause=exc,
            )
            return _failure_acknowledgement(
                request_id=resolved_request_id,
                identity=identity,
                snapshot_digest=snapshot_digest,
                failure_code=code,
                failure_digest=digest,
                acknowledged_ranges=schedule.acknowledgement_cursor_ranges,
            )

    if accepted_metrics_state is not None:
        setattr(
            runner,
            METRICS_CONTINUATION_STATE_ATTRIBUTE,
            accepted_metrics_state,
        )
    return {
        "api_version": 1,
        "request_id": resolved_request_id,
        "expected_controller_id": str(identity["controller_id"]),
        "expected_owner_epoch": int(identity["owner_epoch"]),
        "snapshot_id": str(identity["snapshot_id"]),
        "snapshot_digest_sha256": snapshot_digest,
        "outcome": "success",
        "artifact_manifest_digest_sha256": manifest_digest,
        "acknowledged_channel_cursor_ranges": (
            schedule.acknowledgement_cursor_ranges
        ),
        "failure_code": None,
    }


def _zero_death_age_histogram(runner: Any) -> str:
    survival_edges = list(
        getattr(runner.env_cfg, "tier4_survival_bins_steps", []) or []
    )
    return json.dumps([0 for _ in range(max(0, len(survival_edges) - 1))])


def _log_prepared_metrics_boundary(
    logger: Any,
    runner: Any,
    boundary: _PreparedBoundary,
) -> None:
    counts = boundary.counts
    cohort_rate = float(boundary.cohort_survivors) / float(
        max(1, boundary.cohort_size)
    )
    behavior_total = float(max(1, counts["behavior_action_total"]))
    logger.log_generation(
        generation=boundary.generation,
        agents=[],
        selected_parent_agents=[],
        deaths_this_gen=counts["deaths"],
        successful_births=counts["successful_births"],
        failed_births=counts["failed_births"],
        injury_blocked_births=counts["injury_blocked_births"],
        cohort_size=boundary.cohort_size,
        cohort_survivors=boundary.cohort_survivors,
        cohort_survival_rate=cohort_rate,
        shock_active=0,
        peak_injury=counts["peak_injury"],
        birth_attempts=counts["birth_attempts"],
        birth_pre_cooldown=counts["birth_pre_cooldown"],
        birth_pre_no_energy=counts["birth_pre_no_energy"],
        birth_pre_injury_gate=counts["birth_pre_injury_gate"],
        birth_pre_no_space=counts["birth_pre_no_space"],
        birth_pre_no_mate=counts["birth_pre_no_mate"],
        birth_fail_no_space=counts["birth_fail_no_space"],
        birth_fail_no_mate=counts["birth_fail_no_mate"],
        birth_fail_buffer=counts["birth_fail_buffer"],
        birth_fail_injury_gate=counts["birth_fail_injury_gate"],
        birth_fail_scar_gate=counts["birth_fail_scar_gate"],
        birth_fail_conception=counts["birth_fail_conception"],
        birth_fail_gestation=counts["birth_fail_gestation"],
        tier4_deaths_age_hist_json=_zero_death_age_histogram(runner),
        behavior_proxy_metrics={
            "behavior_action_total": float(counts["behavior_action_total"]),
            "behavior_move_total": float(counts["behavior_move_total"]),
            "behavior_move_n_total": float(counts["behavior_move_n_total"]),
            "behavior_move_s_total": float(counts["behavior_move_s_total"]),
            "behavior_move_e_total": float(counts["behavior_move_e_total"]),
            "behavior_move_w_total": float(counts["behavior_move_w_total"]),
            "behavior_gather_total": float(counts["behavior_gather_total"]),
            "behavior_rest_total": float(counts["behavior_rest_total"]),
            "behavior_attack_total": float(counts["behavior_attack_total"]),
            "behavior_move_fraction": float(counts["behavior_move_total"])
            / behavior_total,
            "behavior_gather_fraction": float(counts["behavior_gather_total"])
            / behavior_total,
            "behavior_rest_fraction": float(counts["behavior_rest_total"])
            / behavior_total,
            "behavior_attack_fraction": float(counts["behavior_attack_total"])
            / behavior_total,
        },
        native_public_output_counts=counts,
        native_public_metrics_overlay=boundary.overlay,
        policy_cache_public_counts=counts,
        sim_cfg=runner.sim_cfg,
        env_cfg=runner.env_cfg,
    )


def _current_metrics_continuation_state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, METRICS_CONTINUATION_STATE_ATTRIBUTE, None)
    if state is None:
        export_state = getattr(getattr(runner, "logger", None), "export_state", None)
        if not callable(export_state):
            raise _fail("runner.logger.export_state is required for checkpoint continuation")
        state = export_state()
    if not isinstance(state, Mapping):
        raise _fail("Architecture B metrics continuation state is malformed")
    try:
        return json.loads(json.dumps(dict(state), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail(f"Architecture B metrics continuation state is not canonical JSON: {exc}") from exc


def _advance_metrics_continuation_state(
    runner: Any,
    boundary: _PreparedBoundary,
    *,
    temporary_parent: Path,
) -> dict[str, Any]:
    from simulation.metrics_pkg.metrics_logger import MetricsLogger

    previous = _current_metrics_continuation_state(runner)
    with tempfile.TemporaryDirectory(
        prefix=".architecture-b-metrics-continuation-",
        dir=temporary_parent,
    ) as temporary_directory:
        logger = MetricsLogger(temporary_directory)
        logger.restore_state(previous)
        _log_prepared_metrics_boundary(logger, runner, boundary)
        state = logger.export_state()
    if not isinstance(state, Mapping):
        raise _fail("detached metrics continuation logger returned malformed state")
    try:
        return json.loads(json.dumps(dict(state), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail(f"detached metrics continuation state is not canonical JSON: {exc}") from exc


def _materialize_native_terminal_chronicle(
    runner: Any,
    prepared: Sequence[_PreparedBoundary],
) -> dict[str, Any] | None:
    recorder = getattr(runner, "scientific_chronicle", None)
    if recorder is None:
        return None
    summaries: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    include_lifecycle = str(getattr(recorder.config, "profile", "notable")) in {
        "scientific_full",
        "debug_trace",
    }
    for boundary in prepared:
        values = dict(boundary.overlay["values"])
        lineage_input = values[NATIVE_LINEAGE_METRICS_INPUT_KEY]
        lineage_counts: dict[str, int] = {}
        for record in lineage_input["records"]:
            founder_id = int(record["founder_id"])
            lineage_id = f"L{founder_id:08d}"
            lineage_counts[lineage_id] = lineage_counts.get(lineage_id, 0) + 1
        counts = boundary.counts
        summaries.append(
            {
                "generation": boundary.generation,
                "alive": counts["alive"],
                "births": counts["successful_births"],
                "failed_births": counts["failed_births"],
                "deaths": counts["deaths"],
                "event_cursor": boundary.lifecycle_end_cursor,
                "birth_attempts": counts["birth_attempts"],
                "capacity": counts["capacity"],
                "free_slots": counts["free_slots"],
                "cohort_size": boundary.cohort_size,
                "cohort_survival_fraction": (
                    float(boundary.cohort_survivors)
                    / float(max(1, boundary.cohort_size))
                ),
                "patch_metrics": {
                    name: values[name]
                    for name in _CHRONICLE_METRIC_FIELDS
                    if name in values
                },
                "native_lineage_counts": lineage_counts,
                "chronicle_source_component": (
                    "native_runtime.architecture_b_reporting"
                ),
                "chronicle_source_record_ref": (
                    f"native_boundary_snapshot:generation={boundary.generation}"
                ),
            }
        )
        if include_lifecycle:
            lifecycle_rows.extend(
                typed_lifecycle_events_to_rows(
                    boundary.lifecycle_events,
                    start=boundary.lifecycle_start_cursor,
                    stop=boundary.lifecycle_end_cursor,
                )
            )
    try:
        recorder.observe_native_terminal_summaries(
            summaries,
            typed_lifecycle_rows=lifecycle_rows,
        )
        recorder.finalize(status="complete")
        if not recorder.finalized:
            raise _fail("native run_many Chronicle did not finalize")
        return {
            "status": "complete",
            "source": "native_terminal_generation_boundary_projections",
            "generation_count": len(summaries),
        }
    except Exception as exc:
        recorder.disabled_after_write_failure = True
        error = f"{type(exc).__name__}: {exc}"
        setattr(runner, "_scientific_chronicle_last_error", error)
        return {
            "status": "failed",
            "source": "native_terminal_generation_boundary_projections",
            "error": error,
        }


def materialize_architecture_b_public_outputs(
    runner: Any,
    run_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize exact public files from detached ``run_many`` snapshots only."""

    prepared, run_dir = _prepare_fanout(runner, run_result)
    logger = runner.logger
    viz = getattr(runner, "viz", None)
    viz_enabled = viz is not None and bool(getattr(viz, "enabled", False))
    if viz_enabled:
        try:
            viz.initialize_run(
                start_generation=prepared[0].generation,
                target_generation=prepared[-1].generation + 1,
                run_kind=(
                    "continuation"
                    if bool(getattr(runner, "_is_resume_run", False))
                    else "base"
                ),
            )
        except Exception as exc:
            raise _fail(f"viz initialization failed: {exc}") from exc
    behavior_summaries: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    lifecycle_summaries: list[dict[str, Any]] = []
    metrics_dir = run_dir / "metrics"
    jsonl_path = metrics_dir / "typed_lifecycle_events.jsonl"
    csv_path = metrics_dir / "typed_lifecycle_events.csv"
    for boundary in prepared:
        try:
            behavior_summary = write_native_behavior_public_outputs(
                runner, boundary.raw
            )
            if not bool(behavior_summary.get("flush_complete", False)):
                raise _fail(
                    f"generation {boundary.generation} behavior output did not flush"
                )
            counts = boundary.counts
            _log_prepared_metrics_boundary(logger, runner, boundary)
            if viz_enabled:
                viz.append_metrics_summary(
                    generation=boundary.generation,
                    alive=counts["alive"],
                    births=counts["successful_births"],
                    deaths=counts["deaths"],
                    avg_gen_time_s=(boundary.boundary_wall_time_ns / 1_000_000_000.0),
                )
            lifecycle_output = write_typed_lifecycle_public_outputs(
                jsonl_path,
                csv_path,
                boundary.lifecycle_events,
                cursor=boundary.lifecycle_start_cursor,
            )
        except ArchitectureBReportingError:
            raise
        except Exception as exc:
            raise _fail(
                f"generation {boundary.generation} public-output persistence failed: {exc}"
            ) from exc

        if int(lifecycle_output["end_cursor"]) != boundary.lifecycle_end_cursor:
            raise _fail(
                f"generation {boundary.generation} lifecycle writer cursor mismatch"
            )
        lifecycle_summary = {
            **dict(lifecycle_output),
            "jsonl_path": str(jsonl_path),
            "csv_path": str(csv_path),
            "total_event_rows": boundary.lifecycle_end_cursor,
            "typed_lifecycle_event_digest": typed_lifecycle_event_digest(
                boundary.lifecycle_events
            ),
            "flush_complete": bool(jsonl_path.is_file() and csv_path.is_file()),
        }
        if not lifecycle_summary["flush_complete"]:
            raise _fail(
                f"generation {boundary.generation} lifecycle outputs did not flush"
            )
        behavior_summaries.append(dict(behavior_summary))
        lifecycle_summaries.append(lifecycle_summary)
        generation_summaries.append(
            {
                "generation": boundary.generation,
                "snapshot_only": True,
                "metrics_overlay_field_count": len(boundary.overlay["values"]),
                "lineage_input_record_count": int(
                    boundary.overlay["values"][NATIVE_LINEAGE_METRICS_INPUT_KEY][
                        "record_count"
                    ]
                ),
                "public_output_counts": dict(counts),
            }
        )

    if viz_enabled:
        try:
            next_generation = prepared[-1].generation + 1
            viz.write_run_status(
                status="complete",
                current_generation=next_generation,
                target_generation=next_generation,
                last_error=None,
                branch_state="COMPLETE",
                run_kind=(
                    "continuation"
                    if bool(getattr(runner, "_is_resume_run", False))
                    else "base"
                ),
            )
        except Exception as exc:
            raise _fail(f"viz completion status persistence failed: {exc}") from exc

    chronicle_materialization = _materialize_native_terminal_chronicle(
        runner,
        prepared,
    )
    final_projection = _mapping(
        run_result["final_checkpoint_projection"],
        path="run_result.final_checkpoint_projection",
    )
    canopy_materialization = materialize_default_terminal_canopy(
        run_root=run_dir,
        run_id=str(run_result["run_id"]),
        terminal_generation=prepared[-1].generation + 1,
        projection=final_projection,
        source_kind="native_run_many_terminal_identity_snapshot",
        source_schema_version="native_run_many_final_checkpoint_projection",
    )
    chronicle_fields = (
        {"chronicle_materialization": chronicle_materialization}
        if chronicle_materialization is not None
        else {}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "detached_architecture_b_run_many_boundaries",
        "resident_owner_read_after_run_many": False,
        "generation_count": len(prepared),
        "behavior_outputs": behavior_summaries,
        "generation_outputs": generation_summaries,
        "typed_lifecycle_outputs": lifecycle_summaries,
        "typed_lifecycle_initial_cursor": prepared[0].lifecycle_start_cursor,
        "typed_lifecycle_final_cursor": prepared[-1].lifecycle_end_cursor,
        "run_dir": str(run_dir),
        "default_canopy": canopy_materialization,
        **chronicle_fields,
    }


__all__ = (
    "ArchitectureBReportingError",
    "ArchitectureBSnapshotPersistenceAcknowledgement",
    "SCHEMA_VERSION",
    "SNAPSHOT_ARTIFACT_MANIFEST_NAME",
    "SNAPSHOT_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "SNAPSHOT_COMPLETION_MARKER_NAME",
    "SNAPSHOT_COMPLETION_SCHEMA_VERSION",
    "SNAPSHOT_STAGING_DIRECTORY",
    "materialize_architecture_b_public_outputs",
    "persist_architecture_b_snapshot",
)
