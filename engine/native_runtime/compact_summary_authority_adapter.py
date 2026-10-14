"""Adapt accepted native runtime authorities to the compact CPU producer.

This module is deliberately projection-only.  It consumes detached generation
and boundary objects, never reads the live owner, and never advances RNG state.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
    checkpoint_source_projection_digest_manifest,
)
from .compact_summary_cpu import (
    SOURCE_SCHEMA_VERSION,
    CompactSummarySourceEnvelope,
    build_compact_summary_row,
)
from .lifecycle_events import (
    LifecycleEventType,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
    validated_typed_lifecycle_event_cursor,
)
from .public_metrics_overlay import NATIVE_LINEAGE_METRICS_INPUT_KEY
from .types import NativeGenerationResult


RUN_RESULT_SCHEMA_VERSION: Final[str] = "eiel_native_architecture_b_run_result_v1"
BOUNDARY_SCHEMA_VERSION: Final[str] = "eiel_native_architecture_b_boundary_v1"
PUBLIC_OVERLAY_SCHEMA_VERSION: Final[str] = "eiel_native_public_metrics_overlay_v1"

_GENERATION_METRIC_SOURCES: Final[dict[str, str]] = {
    "capacity": "capacity",
    "free_slots": "free_slots",
    "birth_attempts": "birth_attempt_rows",
    "native_failed_birth_event_rows": "failed_birth_rows",
    "injury_blocked_births": "injury_blocked_births",
    "moved": "moved",
    "gathered": "gathered",
    "movement_attempts": "movement_attempts",
    "movement_blocked_occupied": "movement_blocked_occupied",
    "movement_blocked_bounds": "movement_blocked_bounds",
    "peak_injury": "peak_injury",
    "policy_loop_tick_calls": "policy_loop_tick_calls",
    "policy_refill_calls": "policy_refill_calls",
    "policy_refill_rows": "policy_refill_rows",
    "policy_cache_hit_calls": "policy_cache_hit_calls",
    "policy_cache_rows_requested": "policy_cache_rows_requested",
    "policy_cache_rows_hit": "policy_cache_rows_hit",
    "policy_cache_rows_missed": "policy_cache_rows_missed",
    "policy_cache_rows_refilled": "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks": "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks": "policy_cache_partial_refill_ticks",
}
_PATCH_METRIC_FIELDS: Final[tuple[str, ...]] = (
    "patch_movement_energy_event_count",
    "patch_spawn_sync_rows",
    "typed_death_event_rows",
    "env_rng_padding_draws",
    "patch_food_harvest_total",
    "patch_scent_emit_total",
    "patch_waste_emit_total",
    "patch_movement_energy_total",
    "energy_collected_delta",
    "patch_scent_move_queued_total",
    "patch_waste_move_queued_total",
    "patch_scent_gather_queued_total",
    "patch_waste_gather_queued_total",
    "patch_movement_energy_extra_total",
    "patch_scent_emitted_total",
    "patch_waste_emitted_total",
    "patch_food_depleted_frac",
    "patch_food_full_frac",
    "patch_density_hotspot_frac",
)
_PATCH_MOVEMENT_ENERGY_TOTAL: Final[str] = "patch_movement_energy_total"
_PATCH_MOVEMENT_ENERGY_ALIAS: Final[str] = "patch_movement_energy_extra_total"
_LINEAGE_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "active_lineages",
    "dominant_lineage_id",
    "dominant_lineage_population_share",
    "top3_lineage_population_share",
    "lineage_population_entropy",
    "lineage_reproduction_entropy",
    "dominant_lineage_reproduction_share",
    "rare_persistent_lineages",
    "lineage_collapse_warning",
    "lineage_collapse_streak",
)
_EVENT_COUNT_NAMES: Final[dict[int, str]] = {
    int(LifecycleEventType.BIRTH_COMMITTED): "birth_committed",
    int(LifecycleEventType.MUTATION): "mutation",
    int(LifecycleEventType.BIRTH_FAILED): "birth_failed",
    int(LifecycleEventType.DEATH_COMMITTED): "death_committed",
}


class CompactSummaryAuthorityAdapterError(ValueError):
    """A required accepted runtime authority is absent or contradictory."""


@dataclass(frozen=True, slots=True)
class CompactSummaryAuthorityInputs:
    """Detached authorities required to build one compact summary source."""

    generation_result: NativeGenerationResult | None
    run_result: Mapping[str, Any]
    boundary_index: int
    public_metrics_overlay: Mapping[str, Any]
    lineage_state_before: Any
    sim_cfg: Any
    policy_mutation_tape: Mapping[str, Any] | None


def _fail(path: str, message: str) -> None:
    raise CompactSummaryAuthorityAdapterError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    return value


def _required(source: Mapping[str, Any], field: str, *, path: str) -> Any:
    if field not in source:
        _fail(f"{path}.{field}", "required authority is absent")
    return source[field]


def _strict_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an int >= {minimum}")
    return value


def _public_count(value: Any, *, path: str) -> int:
    if type(value) is int:
        return _strict_int(value, path=path)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return _strict_int(int(value), path=path)
    _fail(path, "must be an int or an exactly integral finite float >= 0")


def _strict_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a bool")
    return value


def _patch_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    path = "generation_result.raw_result"
    patch = {
        field: _required(raw, field, path=path)
        for field in _PATCH_METRIC_FIELDS
        if field != _PATCH_MOVEMENT_ENERGY_ALIAS
    }
    total = patch[_PATCH_MOVEMENT_ENERGY_TOTAL]
    if type(total) is not float or not math.isfinite(total):
        _fail(f"{path}.{_PATCH_MOVEMENT_ENERGY_TOTAL}", "must be a finite float")
    if _PATCH_MOVEMENT_ENERGY_ALIAS in raw:
        alias = raw[_PATCH_MOVEMENT_ENERGY_ALIAS]
        if (
            type(alias) is not float
            or not math.isfinite(alias)
            or alias.hex() != total.hex()
        ):
            _fail(
                f"{path}.{_PATCH_MOVEMENT_ENERGY_ALIAS}",
                f"must exactly alias {_PATCH_MOVEMENT_ENERGY_TOTAL}",
            )
    patch[_PATCH_MOVEMENT_ENERGY_ALIAS] = total
    return patch


def _tensor_numpy(value: Any, *, path: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if not hasattr(value, "detach") or not hasattr(value, "to"):
        _fail(path, "must be a tensor or NumPy array")
    return np.ascontiguousarray(value.detach().to("cpu").contiguous().numpy())


def _selected_boundary(
    run_result: Mapping[str, Any], boundary_index: int
) -> tuple[Mapping[str, Any], int]:
    if run_result.get("schema_version") != RUN_RESULT_SCHEMA_VERSION:
        _fail("run_result.schema_version", f"must equal {RUN_RESULT_SCHEMA_VERSION!r}")
    if type(boundary_index) is not int or boundary_index < 0:
        _fail("boundary_index", "must be a non-negative int")
    boundaries = _required(run_result, "boundaries", path="run_result")
    if isinstance(boundaries, (str, bytes, bytearray)) or not isinstance(
        boundaries, Sequence
    ):
        _fail("run_result.boundaries", "must be a sequence")
    if boundary_index >= len(boundaries):
        _fail("boundary_index", f"is outside {len(boundaries)} boundaries")
    boundary = _mapping(boundaries[boundary_index], path=f"run_result.boundaries[{boundary_index}]")
    if boundary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        _fail(
            f"run_result.boundaries[{boundary_index}].schema_version",
            f"must equal {BOUNDARY_SCHEMA_VERSION!r}",
        )
    return boundary, len(boundaries)


def _validated_generation_result(
    result: NativeGenerationResult | None,
    boundary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int, int, str]:
    boundary_raw = _mapping(
        _required(boundary, "generation_result", path="boundary"),
        path="boundary.generation_result",
    )
    generation = _strict_int(
        _required(boundary_raw, "generation_index", path="boundary.generation_result"),
        path="boundary.generation_result.generation_index",
    )
    steps = _strict_int(
        _required(boundary_raw, "steps_per_generation", path="boundary.generation_result"),
        path="boundary.generation_result.steps_per_generation",
        minimum=1,
    )
    if _strict_int(
        _required(boundary_raw, "native_tick_loop_count", path="boundary.generation_result"),
        path="boundary.generation_result.native_tick_loop_count",
        minimum=1,
    ) != steps:
        _fail("boundary.generation_result.native_tick_loop_count", "must equal steps_per_generation")
    for field in ("executed", "hidden_fallback", "native_owner", "single_native_generation_call"):
        expected = field != "hidden_fallback"
        if _required(boundary_raw, field, path="boundary.generation_result") is not expected:
            _fail(f"boundary.generation_result.{field}", f"must be {expected}")
    if boundary.get("completed_generation") != generation:
        _fail("boundary.completed_generation", "contradicts generation result")
    if boundary.get("next_generation") != generation + 1:
        _fail("boundary.next_generation", "must be completed_generation + 1")
    if boundary.get("tick") != 0 or boundary.get("phase") != "generation_complete":
        _fail("boundary", "must be a complete generation boundary at tick zero")
    reporting = _mapping(
        _required(boundary, "generation_end_reporting_frame", path="boundary"),
        path="boundary.generation_end_reporting_frame",
    )
    metadata = _mapping(
        _required(reporting, "metadata", path="boundary.generation_end_reporting_frame"),
        path="boundary.generation_end_reporting_frame.metadata",
    )
    world = metadata.get("world")
    if world not in {"flat", "patch"}:
        _fail("boundary.generation_end_reporting_frame.metadata.world", "must be 'flat' or 'patch'")
    if result is not None:
        if not isinstance(result, NativeGenerationResult):
            _fail("generation_result", "must be NativeGenerationResult or None")
        if result.schema_version != "eiel_native_generation_result_v1":
            _fail("generation_result.schema_version", "unsupported native generation result")
        if result.executed is not True:
            _fail("generation_result.executed", "must be true")
        scalar_matches = {
            "generation_index": result.generation_index,
            "steps_per_generation": result.steps_per_generation,
            "native_tick_loop_count": result.native_tick_loop_count,
            "committed_birth_rows": result.committed_birth_rows,
        }
        raw = _mapping(result.raw_result, path="generation_result.raw_result")
        for field, expected in scalar_matches.items():
            if _required(raw, field, path="generation_result.raw_result") != expected:
                _fail(f"generation_result.raw_result.{field}", "contradicts wrapper")
            if _required(boundary_raw, field, path="boundary.generation_result") != expected:
                _fail(f"boundary.generation_result.{field}", "contradicts wrapper")
        if result.world != world:
            _fail("generation_result.world", "contradicts reporting boundary")
        if boundary_raw.get("behavior_rows") != raw.get("behavior_rows"):
            _fail("boundary.generation_result.behavior_rows", "contradicts native wrapper")
    return boundary_raw, generation, steps, str(world)


def _projection(
    boundary: Mapping[str, Any], field: str, *, generation: int
) -> Mapping[str, Any]:
    projection = _mapping(_required(boundary, field, path="boundary"), path=f"boundary.{field}")
    metadata = _mapping(_required(projection, "metadata", path=f"boundary.{field}"), path=f"boundary.{field}.metadata")
    if metadata.get("generation") != generation + 1:
        _fail(f"boundary.{field}.metadata.generation", "must be next generation")
    if metadata.get("tick") != 0:
        _fail(f"boundary.{field}.metadata.tick", "must be zero")
    if metadata.get("agent_grid_runner_phase") != "generation_boundary":
        _fail(f"boundary.{field}.metadata.agent_grid_runner_phase", "must be generation_boundary")
    return projection


def _event_counts(event_types: np.ndarray, start: int, stop: int) -> dict[str, int]:
    values = event_types[start:stop]
    unknown = sorted(set(int(item) for item in values.tolist()) - set(_EVENT_COUNT_NAMES))
    if unknown:
        _fail("typed_lifecycle.event_type", f"contains unknown codes {unknown}")
    counts = {name: 0 for name in _EVENT_COUNT_NAMES.values()}
    for code, name in _EVENT_COUNT_NAMES.items():
        counts[name] = int(np.count_nonzero(values == code))
    return counts


def _lifecycle_authority(
    raw: Mapping[str, Any],
    boundary: Mapping[str, Any],
    reporting_projection: Mapping[str, Any],
    post_projection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int], dict[str, int]]:
    reporting_events = typed_lifecycle_events_from_projection(reporting_projection)
    post_events = typed_lifecycle_events_from_projection(post_projection)
    reporting_cursor = validated_typed_lifecycle_event_cursor(reporting_projection, reporting_events)
    end_cursor = validated_typed_lifecycle_event_cursor(post_projection, post_events)
    generation_rows = _strict_int(
        _required(raw, "typed_event_rows", path="generation_result.raw_result"),
        path="generation_result.raw_result.typed_event_rows",
    )
    turnover = _mapping(_required(boundary, "turnover_result", path="boundary"), path="boundary.turnover_result")
    turnover_rows = _strict_int(
        _required(turnover, "typed_event_rows", path="boundary.turnover_result"),
        path="boundary.turnover_result.typed_event_rows",
    )
    start_cursor = reporting_cursor - generation_rows
    if start_cursor < 0:
        _fail("typed_lifecycle", "generation delta exceeds the reporting cursor")
    if end_cursor - reporting_cursor != turnover_rows:
        _fail("boundary.turnover_result.typed_event_rows", "contradicts post-turnover cursor")
    event_types = _tensor_numpy(post_events["event_type"], path="typed_lifecycle.event_type")
    reason_codes = _tensor_numpy(post_events["reason_code"], path="typed_lifecycle.reason_code")
    generation_counts = _event_counts(event_types, start_cursor, reporting_cursor)
    turnover_counts = _event_counts(event_types, reporting_cursor, end_cursor)
    death_mask = event_types[start_cursor:reporting_cursor] == int(LifecycleEventType.DEATH_COMMITTED)
    death_reasons = reason_codes[start_cursor:reporting_cursor][death_mask]
    unknown_deaths = sorted(set(int(item) for item in death_reasons.tolist()) - {1, 2})
    if unknown_deaths:
        _fail("typed_lifecycle.reason_code", f"contains unknown committed-death codes {unknown_deaths}")
    deaths_by_class = {
        "energy_depleted": int(np.count_nonzero(death_reasons == 1)),
        "health_depleted": int(np.count_nonzero(death_reasons == 2)),
    }
    if sum(deaths_by_class.values()) != _strict_int(
        _required(raw, "death_rows", path="generation_result.raw_result"),
        path="generation_result.raw_result.death_rows",
    ):
        _fail("typed_lifecycle.death_rows", "contradicts generation_result.death_rows")
    event_cursor = _strict_int(
        _required(post_projection, "event_cursor", path="post_turnover_checkpoint_projection"),
        path="post_turnover_checkpoint_projection.event_cursor",
    )
    if event_cursor != end_cursor:
        _fail("post_turnover_checkpoint_projection.event_cursor", "contradicts typed lifecycle cursor")
    event_authority = {
        "event_cursor_start": start_cursor,
        "event_cursor_pre_turnover": reporting_cursor,
        "event_cursor_end": end_cursor,
        "authoritative_event_cursor": event_cursor,
        "event_row_count": end_cursor - start_cursor,
        "generation_event_count": generation_rows,
        "turnover_event_count": turnover_rows,
    }
    return event_authority, generation_counts, turnover_counts, deaths_by_class


def _overlay_values(overlay: Mapping[str, Any]) -> Mapping[str, Any]:
    source = _mapping(overlay, path="public_metrics_overlay")
    if source.get("schema_version") != PUBLIC_OVERLAY_SCHEMA_VERSION:
        _fail("public_metrics_overlay.schema_version", f"must equal {PUBLIC_OVERLAY_SCHEMA_VERSION!r}")
    if source.get("source") != "native_owner_checkpoint_projection":
        _fail("public_metrics_overlay.source", "must be native_owner_checkpoint_projection")
    return _mapping(_required(source, "values", path="public_metrics_overlay"), path="public_metrics_overlay.values")


def _behavior_authority(raw: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    from .flat_population_reporting import validated_native_flat_behavior_rows

    generation = _strict_int(
        _required(raw, "generation_index", path="generation_result.raw_result"),
        path="generation_result.raw_result.generation_index",
    )
    rows = tuple(
        dict(row)
        for row in validated_native_flat_behavior_rows(
            _required(raw, "behavior_rows", path="generation_result.raw_result"),
            expected_generation=generation,
        )
    )
    cohort_size = sum(row["cohort"] == "start" for row in rows)
    cohort_survivors = sum(
        row["cohort"] == "start" and int(row["alive_end"]) == 1 for row in rows
    )
    names = {
        "action_total": "behavior_action_total",
        "move_total": "behavior_move_total",
        "move_n": "behavior_move_n_total",
        "move_s": "behavior_move_s_total",
        "move_e": "behavior_move_e_total",
        "move_w": "behavior_move_w_total",
        "gather": "behavior_gather_total",
        "rest": "behavior_rest_total",
        "attack": "behavior_attack_total",
    }
    out = {
        "rows": rows,
        "row_count": len(rows),
        "cohort_size": cohort_size,
        "cohort_survivors": cohort_survivors,
    }
    for target, source in names.items():
        out[target] = _public_count(
            _required(values, source, path="public_metrics_overlay.values"),
            path=f"public_metrics_overlay.values.{source}",
        )
    return out


def _lineage_authority(
    values: Mapping[str, Any],
    *,
    generation: int,
    state_before: Any,
    sim_cfg: Any,
) -> dict[str, Any]:
    from simulation.metrics_pkg.lineage import (
        LineageMetricsState,
        compute_lineage_snapshot,
        native_lineage_agents_from_input,
    )

    if not isinstance(state_before, LineageMetricsState):
        _fail("lineage_state_before", "must be LineageMetricsState")
    payload = _mapping(
        _required(values, NATIVE_LINEAGE_METRICS_INPUT_KEY, path="public_metrics_overlay.values"),
        path=f"public_metrics_overlay.values.{NATIVE_LINEAGE_METRICS_INPUT_KEY}",
    )
    agents = native_lineage_agents_from_input(payload, expected_generation=generation)
    cloned_state = LineageMetricsState.from_dict(copy.deepcopy(state_before.to_dict()))
    summary, _rows, _events, _state_after = compute_lineage_snapshot(
        generation=generation,
        agents=agents,
        sim_cfg=sim_cfg,
        state=cloned_state,
    )
    records = _required(payload, "records", path=NATIVE_LINEAGE_METRICS_INPUT_KEY)
    record_count = _strict_int(
        _required(payload, "record_count", path=NATIVE_LINEAGE_METRICS_INPUT_KEY),
        path=f"{NATIVE_LINEAGE_METRICS_INPUT_KEY}.record_count",
    )
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        _fail(f"{NATIVE_LINEAGE_METRICS_INPUT_KEY}.records", "must be a sequence")
    if record_count != len(records):
        _fail(f"{NATIVE_LINEAGE_METRICS_INPUT_KEY}.record_count", "contradicts records")
    out = {
        "generation": generation,
        "records": tuple(copy.deepcopy(dict(row)) for row in records),
        "record_count": record_count,
    }
    for field in _LINEAGE_SUMMARY_FIELDS:
        out[field] = _required(summary, field, path="lineage_summary")
    warning = out["lineage_collapse_warning"]
    if type(warning) is int and warning in {0, 1}:
        out["lineage_collapse_warning"] = bool(warning)
    if out["dominant_lineage_id"] == "":
        out["dominant_lineage_id"] = None
    return out


def _settled_authority(
    run_result: Mapping[str, Any], *, boundary_index: int, boundary_count: int
) -> dict[str, Any]:
    terminal_status = _required(run_result, "terminal_status", path="run_result")
    is_final = boundary_index == boundary_count - 1
    if terminal_status == "failed":
        _fail(
            "run_result.root_failure.failure_stage",
            "the current failed-run contract has no compact failure-stage authority",
        )
    if terminal_status not in {"run_complete", "cancelled"}:
        _fail("run_result.terminal_status", "must be run_complete or cancelled")
    status = terminal_status if is_final else "not_terminal"
    if status == "cancelled":
        if run_result.get("cancelled_after_completed_generation") is not True:
            _fail("run_result.cancelled_after_completed_generation", "must be true")
    return {
        "summary_valid": True,
        "summary_status": "complete",
        "terminal": {
            "status": status,
            "recoverable_error_status": "none",
            "recoverable_error_code": None,
            "recoverable_error_details": None,
            "sticky_failure": False,
            "failure_generation": None,
            "failure_tick": None,
            "failure_stage": None,
        },
    }


def _schedule_authority(boundary: Mapping[str, Any], *, cancelled: bool) -> dict[str, Any]:
    state = _mapping(_required(boundary, "schedule_state", path="boundary"), path="boundary.schedule_state")
    if state.get("schema_version") != "eiel_native_snapshot_schedule_v1":
        _fail("boundary.schedule_state.schema_version", "unsupported schedule state")
    explicit = _required(state, "explicit_next_generations", path="boundary.schedule_state")
    if type(explicit) is not list:
        _fail("boundary.schedule_state.explicit_next_generations", "must be a list")
    return {
        "schedule_origin_generation": _strict_int(
            _required(state, "schedule_origin_generation", path="boundary.schedule_state"),
            path="boundary.schedule_state.schedule_origin_generation",
        ),
        "explicit_next_generations": list(explicit),
        "cancellation_due": cancelled,
    }


def _checkpoint_authorities(post_projection: Mapping[str, Any], boundary: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    source = checkpoint_source_projection_digest_manifest(post_projection)
    rng = _mapping(_required(source, "rng", path="source_projection_manifest"), path="source_projection_manifest.rng")
    if rng.get("complete") is not True:
        _fail(
            "post_turnover_checkpoint_projection.rng",
            f"authority is incomplete: {rng.get('missing_required_metadata_fields')!r}",
        )
    trajectory = checkpoint_digest_manifest(post_projection, layer=DigestLayer.TRAJECTORY_STATE)
    storage = checkpoint_digest_manifest(post_projection, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE)
    recorded = _mapping(
        _required(boundary, "checkpoint_projection_digest_manifests", path="boundary"),
        path="boundary.checkpoint_projection_digest_manifests",
    )
    expected = {
        "source_projection": source,
        "trajectory_state": trajectory,
        "canonical_checkpoint_storage": storage,
    }
    for name, manifest in expected.items():
        row = _mapping(_required(recorded, name, path="checkpoint_projection_digest_manifests"), path=f"checkpoint_projection_digest_manifests.{name}")
        if row.get("digest") != manifest.get("digest"):
            _fail(f"checkpoint_projection_digest_manifests.{name}.digest", "contradicts projection")
    return (
        {
            "authoritative_state_digest_sha256": str(trajectory["digest"]),
            "canonical_checkpoint_storage_digest_sha256": str(storage["digest"]),
            "next_generation_start_state_digest_sha256": str(source["digest"]),
        },
        str(rng["digest"]),
    )


def assemble_compact_summary_source(
    inputs: CompactSummaryAuthorityInputs,
) -> CompactSummarySourceEnvelope:
    """Build a strict CPU source envelope from detached accepted authorities."""

    if not isinstance(inputs, CompactSummaryAuthorityInputs):
        _fail("inputs", "must be CompactSummaryAuthorityInputs")
    run_result = _mapping(inputs.run_result, path="run_result")
    boundary, boundary_count = _selected_boundary(run_result, inputs.boundary_index)
    raw, generation, steps_per_generation, world = _validated_generation_result(
        inputs.generation_result, boundary
    )
    reporting = _projection(boundary, "generation_end_reporting_frame", generation=generation)
    post = _projection(boundary, "post_turnover_checkpoint_projection", generation=generation)
    reporting_world = _mapping(reporting["metadata"], path="reporting.metadata").get("world")
    post_world = _mapping(post["metadata"], path="post.metadata").get("world")
    if reporting_world != world or post_world != reporting_world:
        _fail("boundary.projection.metadata.world", "contradicts generation result")

    values = _overlay_values(inputs.public_metrics_overlay)
    event_authority, generation_counts, turnover_counts, deaths_by_class = _lifecycle_authority(
        raw, boundary, reporting, post
    )
    settled = _settled_authority(
        run_result,
        boundary_index=inputs.boundary_index,
        boundary_count=boundary_count,
    )
    schedule = _schedule_authority(
        boundary,
        cancelled=settled["terminal"]["status"] == "cancelled",
    )
    state_digests, rng_digest = _checkpoint_authorities(post, boundary)
    if inputs.policy_mutation_tape is None:
        _fail(
            "policy_mutation_tape",
            "the current Architecture B boundary does not export the resident consumed-prefix tape",
        )

    post_tensors = _mapping(_required(post, "tensors", path="post_turnover_checkpoint_projection"), path="post_turnover_checkpoint_projection.tensors")
    active_ids = _tensor_numpy(
        _required(post_tensors, "agent_ids_by_alive_order", path="post_turnover_checkpoint_projection.tensors"),
        path="post_turnover_checkpoint_projection.tensors.agent_ids_by_alive_order",
    )
    if active_ids.dtype != np.dtype(np.int64) or active_ids.ndim != 1:
        _fail("post_turnover_checkpoint_projection.tensors.agent_ids_by_alive_order", "must be int64[active]")
    if active_ids.size and (np.any(active_ids < 0) or np.any(active_ids[1:] <= active_ids[:-1])):
        _fail("post_turnover_checkpoint_projection.tensors.agent_ids_by_alive_order", "must be strictly ascending unique non-negative IDs")
    post_metadata = _mapping(post["metadata"], path="post_turnover_checkpoint_projection.metadata")
    alive_after = int(active_ids.size)
    if post_metadata.get("alive_count") != alive_after:
        _fail("post_turnover_checkpoint_projection.metadata.alive_count", "contradicts active IDs")
    capacity = _strict_int(_required(raw, "capacity", path="generation_result.raw_result"), path="generation_result.raw_result.capacity")
    if alive_after > capacity:
        _fail("post_turnover_checkpoint_projection.metadata.alive_count", "exceeds capacity")

    turnover = _mapping(boundary["turnover_result"], path="boundary.turnover_result")
    for field, expected in (("executed", True), ("hidden_fallback", False), ("native_owner", True)):
        if _required(turnover, field, path="boundary.turnover_result") is not expected:
            _fail(f"boundary.turnover_result.{field}", f"must be {expected}")
    if turnover.get("alive_after") != alive_after or turnover.get("next_generation") != generation + 1 or turnover.get("next_tick") != 0:
        _fail("boundary.turnover_result", "contradicts post-turnover boundary")

    generation_metrics = {
        target: _required(raw, source, path="generation_result.raw_result")
        for target, source in _GENERATION_METRIC_SOURCES.items()
    }
    behavior = _behavior_authority(raw, values)
    lineage = _lineage_authority(
        values,
        generation=generation,
        state_before=inputs.lineage_state_before,
        sim_cfg=inputs.sim_cfg,
    )
    patch = None
    if world == "patch":
        patch = _patch_metrics(raw)

    source = CompactSummarySourceEnvelope(
        schema_version=SOURCE_SCHEMA_VERSION,
        pre_turnover={
            "world": world,
            "generation_start": {"generation": generation, "tick": 0},
            "generation_end": {
                "generation": generation,
                "tick_exclusive": steps_per_generation,
            },
            "alive": _required(raw, "alive_count", path="generation_result.raw_result"),
            "births": _required(raw, "committed_birth_rows", path="generation_result.raw_result"),
            "failed_births": _required(raw, "failed_birth_rows", path="generation_result.raw_result"),
            "deaths": _required(raw, "death_rows", path="generation_result.raw_result"),
            "deaths_by_class": deaths_by_class,
            "event_cursor_start": event_authority["event_cursor_start"],
            "event_cursor_pre_turnover": event_authority["event_cursor_pre_turnover"],
            "generation_event_count": event_authority["generation_event_count"],
            "generation_counts_by_type": generation_counts,
            "generation_metrics": generation_metrics,
        },
        post_turnover={
            "next_generation_start": {"generation": generation + 1, "tick": 0},
            "turnover": {"count": 1, "status": "committed"},
            "event_cursor_end": event_authority["event_cursor_end"],
            "authoritative_event_cursor": event_authority["authoritative_event_cursor"],
            "event_row_count": event_authority["event_row_count"],
            "turnover_event_count": event_authority["turnover_event_count"],
            "turnover_counts_by_type": turnover_counts,
            "identity": {
                "population_next_id": _strict_int(
                    _required(post_metadata, "population_next_id", path="post_turnover_checkpoint_projection.metadata"),
                    path="post_turnover_checkpoint_projection.metadata.population_next_id",
                    minimum=1,
                ),
                "active_agent_count": alive_after,
                "active_agent_id_min": None if alive_after == 0 else int(active_ids[0]),
                "active_agent_id_max": None if alive_after == 0 else int(active_ids[-1]),
                "free_slot_count": capacity - alive_after,
            },
            "active_agent_ids": active_ids.copy(),
        },
        settled=settled,
        schedule=schedule,
        patch=patch,
        behavior=behavior,
        lineage=lineage,
        rng={
            "authority_digest_sha256": rng_digest,
            "policy_mutation_tape": inputs.policy_mutation_tape,
            "authority_complete": True,
            "export_non_consuming": True,
        },
        digests={
            "events_active_prefix_digest_sha256": typed_lifecycle_event_digest(
                typed_lifecycle_events_from_projection(post)
            ),
            **state_digests,
        },
    )
    try:
        build_compact_summary_row(source)
    except Exception as exc:
        if isinstance(exc, CompactSummaryAuthorityAdapterError):
            raise
        raise CompactSummaryAuthorityAdapterError(
            f"compact_summary_source: downstream validation failed: {exc}"
        ) from exc
    return source


__all__ = [
    "CompactSummaryAuthorityAdapterError",
    "CompactSummaryAuthorityInputs",
    "assemble_compact_summary_source",
]
