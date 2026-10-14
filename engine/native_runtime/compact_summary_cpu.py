"""Independent CPU producer for compact generation-summary v1 rows.

The producer joins explicit CPU authorities from the completed generation,
post-turnover boundary, and settled result.  It deliberately has no CUDA
input: native planes may be compared with its output, but cannot supply it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .compact_summary_digests import (
    active_agent_ids_digest_sha256,
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
    policy_mutation_tape_digest_sha256,
    recoverable_error_digest_sha256,
)


SOURCE_SCHEMA_VERSION: Final[str] = "eiel_compact_summary_cpu_source_v1"
WIRE_SCHEMA_VERSION: Final[str] = "eiel_compact_generation_summary_v1"
RECOVERABLE_ERROR_SCHEMA_VERSION: Final[str] = (
    "eiel_recoverable_error_digest_v1"
)
MAX_INT64: Final[int] = (1 << 63) - 1

_SOURCE_FIELDS = (
    "schema_version",
    "pre_turnover",
    "post_turnover",
    "settled",
    "schedule",
    "patch",
    "behavior",
    "lineage",
    "rng",
    "digests",
)
_EVENT_COUNT_FIELDS = (
    "birth_committed",
    "mutation",
    "birth_failed",
    "death_committed",
)
_GENERATION_METRIC_FIELDS = (
    "capacity",
    "free_slots",
    "birth_attempts",
    "native_failed_birth_event_rows",
    "injury_blocked_births",
    "moved",
    "gathered",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "peak_injury",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
)
_PATCH_METRIC_FIELDS = (
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
_BEHAVIOR_AUTHORITY_FIELDS = (
    "rows",
    "row_count",
    "cohort_size",
    "cohort_survivors",
    "action_total",
    "move_total",
    "move_n",
    "move_s",
    "move_e",
    "move_w",
    "gather",
    "rest",
    "attack",
)
_LINEAGE_AUTHORITY_FIELDS = (
    "generation",
    "records",
    "record_count",
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
_PRE_TURNOVER_FIELDS = (
    "world",
    "generation_start",
    "generation_end",
    "alive",
    "births",
    "failed_births",
    "deaths",
    "deaths_by_class",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "generation_event_count",
    "generation_counts_by_type",
    "generation_metrics",
)
_POST_TURNOVER_FIELDS = (
    "next_generation_start",
    "turnover",
    "event_cursor_end",
    "authoritative_event_cursor",
    "event_row_count",
    "turnover_event_count",
    "turnover_counts_by_type",
    "identity",
    "active_agent_ids",
)
_IDENTITY_AUTHORITY_FIELDS = (
    "population_next_id",
    "active_agent_count",
    "active_agent_id_min",
    "active_agent_id_max",
    "free_slot_count",
)
_TERMINAL_AUTHORITY_FIELDS = (
    "status",
    "recoverable_error_status",
    "recoverable_error_code",
    "recoverable_error_details",
    "sticky_failure",
    "failure_generation",
    "failure_tick",
    "failure_stage",
)
_SETTLED_FIELDS = ("summary_valid", "summary_status", "terminal")
_SCHEDULE_FIELDS = (
    "schedule_origin_generation",
    "explicit_next_generations",
    "cancellation_due",
)
_RNG_FIELDS = (
    "authority_digest_sha256",
    "policy_mutation_tape",
    "authority_complete",
    "export_non_consuming",
)
_DIGEST_FIELDS = (
    "events_active_prefix_digest_sha256",
    "authoritative_state_digest_sha256",
    "canonical_checkpoint_storage_digest_sha256",
    "next_generation_start_state_digest_sha256",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CompactSummarySourceError(ValueError):
    """A CPU authority envelope is missing, malformed, or contradictory."""


@dataclass(frozen=True, slots=True)
class CompactSummarySourceEnvelope:
    """Typed outer envelope; each authority remains a strict mapping."""

    schema_version: str
    pre_turnover: Mapping[str, Any]
    post_turnover: Mapping[str, Any]
    settled: Mapping[str, Any]
    schedule: Mapping[str, Any]
    patch: Mapping[str, Any] | None
    behavior: Mapping[str, Any]
    lineage: Mapping[str, Any]
    rng: Mapping[str, Any]
    digests: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "CompactSummarySourceEnvelope":
        mapping = _require_exact_mapping(source, _SOURCE_FIELDS, path="source")
        return cls(**{field: mapping[field] for field in _SOURCE_FIELDS})


def _fail(path: str, message: str) -> None:
    raise CompactSummarySourceError(f"{path}: {message}")


def _require_exact_mapping(
    value: Any,
    expected: Sequence[str],
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    if any(type(key) is not str for key in value):
        _fail(path, "field names must be strings")
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        _fail(path, f"fields mismatch: missing={missing} extra={extra}")
    return value


def _strict_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int:
        _fail(path, "must be an int")
    if value < minimum or value > MAX_INT64:
        _fail(path, f"must be in [{minimum}, {MAX_INT64}]")
    return value


def _strict_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a bool")
    return value


def _strict_sha256(value: Any, *, path: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(path, "must be a lowercase SHA-256 hex digest")
    return value


def _copy_exact_object(
    value: Any,
    fields: Sequence[str],
    *,
    path: str,
) -> dict[str, Any]:
    mapping = _require_exact_mapping(value, fields, path=path)
    return {field: mapping[field] for field in fields}


def _strict_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, (Mapping, str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        _fail(path, "must be a sequence")
    return value


def _strict_json_copy(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "must be finite")
        return value
    if type(value) is list:
        return [
            _strict_json_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _fail(path, "JSON object keys must be strings")
        return {
            key: _strict_json_copy(child, path=f"{path}.{key}")
            for key, child in value.items()
        }
    _fail(path, f"unsupported strict JSON type {type(value).__name__}")


def _canonical_json_without_lf(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompactSummarySourceError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def _recoverable_error_digest(terminal: Mapping[str, Any]) -> str | None:
    status = terminal["recoverable_error_status"]
    code = terminal["recoverable_error_code"]
    details = terminal["recoverable_error_details"]
    if status == "none":
        if code is not None or details is not None:
            _fail(
                "source.settled.terminal",
                "absent recoverable error requires null code and details",
            )
        return None
    if status != "present":
        _fail(
            "source.settled.terminal.recoverable_error_status",
            "must be 'none' or 'present'",
        )
    if type(code) is not str or not code or len(code.encode("utf-8")) > 64:
        _fail(
            "source.settled.terminal.recoverable_error_code",
            "must be a non-empty string using at most 64 UTF-8 bytes",
        )
    if not isinstance(details, Mapping):
        _fail(
            "source.settled.terminal.recoverable_error_details",
            "must be a strict JSON object when present",
        )
    strict_details = _strict_json_copy(
        details,
        path="source.settled.terminal.recoverable_error_details",
    )
    return recoverable_error_digest_sha256(code, strict_details)


def _behavior_metrics(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_exact_mapping(
        authority,
        _BEHAVIOR_AUTHORITY_FIELDS,
        path="source.behavior",
    )
    rows = _strict_sequence(source["rows"], path="source.behavior.rows")
    try:
        rows_digest = behavior_rows_digest_sha256(rows)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CompactSummarySourceError(f"source.behavior.rows: {exc}") from exc

    directional = {
        field: sum(row[field] for row in rows)
        for field in ("move_n", "move_s", "move_e", "move_w")
    }
    totals = {
        "row_count": len(rows),
        "cohort_size": sum(row["cohort"] == "start" for row in rows),
        "cohort_survivors": sum(
            row["cohort"] == "start" and row["alive_end"] == 1 for row in rows
        ),
        **directional,
        "gather": sum(row["gather"] for row in rows),
        "rest": sum(row["rest"] for row in rows),
        "attack": sum(row["attack"] for row in rows),
    }
    totals["move_total"] = sum(directional.values())
    totals["action_total"] = (
        totals["move_total"]
        + totals["gather"]
        + totals["rest"]
        + totals["attack"]
    )
    for field, expected in totals.items():
        actual = source[field]
        if type(actual) is not int or actual != expected:
            _fail(
                f"source.behavior.{field}",
                f"contradicts behavior rows: expected {expected!r}",
            )
    return {
        "row_count": source["row_count"],
        "rows_digest_sha256": rows_digest,
        "cohort_size": source["cohort_size"],
        "cohort_survivors": source["cohort_survivors"],
        "action_total": source["action_total"],
        "move_total": source["move_total"],
        "move_n": source["move_n"],
        "move_s": source["move_s"],
        "move_e": source["move_e"],
        "move_w": source["move_w"],
        "gather": source["gather"],
        "rest": source["rest"],
        "attack": source["attack"],
    }


def _lineage_metrics(
    authority: Mapping[str, Any],
    *,
    generation: int,
) -> dict[str, Any]:
    source = _require_exact_mapping(
        authority,
        _LINEAGE_AUTHORITY_FIELDS,
        path="source.lineage",
    )
    lineage_generation = _strict_int(
        source["generation"], path="source.lineage.generation"
    )
    if lineage_generation != generation:
        _fail("source.lineage.generation", "must match the completed generation")
    records = _strict_sequence(source["records"], path="source.lineage.records")
    record_count = _strict_int(
        source["record_count"], path="source.lineage.record_count"
    )
    if record_count != len(records):
        _fail(
            "source.lineage.record_count",
            f"contradicts lineage records: expected {len(records)}",
        )
    try:
        records_digest = lineage_records_digest_sha256(
            lineage_generation, records
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CompactSummarySourceError(f"source.lineage.records: {exc}") from exc
    return {
        "record_count": record_count,
        "records_digest_sha256": records_digest,
        **{
            field: source[field]
            for field in _LINEAGE_AUTHORITY_FIELDS
            if field not in {"generation", "records", "record_count"}
        },
    }


def _active_agent_id_values(value: Any) -> np.ndarray:
    try:
        active_agent_ids_digest_sha256(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CompactSummarySourceError(
            f"source.post_turnover.active_agent_ids: {exc}"
        ) from exc
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return np.ascontiguousarray(value.detach().to("cpu").contiguous().numpy())


def _identity(
    authority: Mapping[str, Any], active_agent_ids: Any
) -> dict[str, Any]:
    identity = _require_exact_mapping(
        authority,
        _IDENTITY_AUTHORITY_FIELDS,
        path="source.post_turnover.identity",
    )
    array = _active_agent_id_values(active_agent_ids)
    count = int(array.size)
    expected_min = None if count == 0 else int(array[0])
    expected_max = None if count == 0 else int(array[-1])
    for field, expected in (
        ("active_agent_count", count),
        ("active_agent_id_min", expected_min),
        ("active_agent_id_max", expected_max),
    ):
        actual = identity[field]
        if type(actual) is not type(expected) or actual != expected:
            _fail(
                f"source.post_turnover.identity.{field}",
                f"contradicts active_agent_ids: expected {expected!r}",
            )
    return {
        **{field: identity[field] for field in _IDENTITY_AUTHORITY_FIELDS},
        "active_agent_ids_digest_sha256": active_agent_ids_digest_sha256(
            active_agent_ids
        ),
    }


def _policy_tape_cursor(tape: Any) -> int:
    if not isinstance(tape, Mapping) or "policy_birth_tape_cursor" not in tape:
        _fail(
            "source.rng.policy_mutation_tape",
            "must contain policy_birth_tape_cursor",
        )
    value = tape["policy_birth_tape_cursor"]
    if type(value) is int:
        return _strict_int(value, path="source.rng.policy_mutation_tape.cursor")
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "to"):
        array = value.detach().to("cpu").contiguous().numpy()
    else:
        _fail(
            "source.rng.policy_mutation_tape.policy_birth_tape_cursor",
            "must be an int or int64[1] array/tensor",
        )
    if array.dtype != np.dtype(np.int64) or array.shape != (1,):
        _fail(
            "source.rng.policy_mutation_tape.policy_birth_tape_cursor",
            "must have dtype int64 and shape [1]",
        )
    return _strict_int(
        int(array[0]),
        path="source.rng.policy_mutation_tape.policy_birth_tape_cursor",
    )


def _rng(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_exact_mapping(authority, _RNG_FIELDS, path="source.rng")
    authority_complete = _strict_bool(
        source["authority_complete"], path="source.rng.authority_complete"
    )
    export_non_consuming = _strict_bool(
        source["export_non_consuming"], path="source.rng.export_non_consuming"
    )
    tape = source["policy_mutation_tape"]
    try:
        tape_digest = policy_mutation_tape_digest_sha256(tape)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CompactSummarySourceError(
            f"source.rng.policy_mutation_tape: {exc}"
        ) from exc
    return {
        "authority_digest_sha256": _strict_sha256(
            source["authority_digest_sha256"],
            path="source.rng.authority_digest_sha256",
        ),
        "policy_mutation_tape_cursor": _policy_tape_cursor(tape),
        "policy_mutation_tape_digest_sha256": tape_digest,
        "authority_complete": authority_complete,
        "export_non_consuming": export_non_consuming,
    }


def _checkpoint_due(
    authority: Mapping[str, Any],
    *,
    next_generation: int,
    terminal_status: Any,
) -> bool:
    schedule = _require_exact_mapping(
        authority, _SCHEDULE_FIELDS, path="source.schedule"
    )
    origin = _strict_int(
        schedule["schedule_origin_generation"],
        path="source.schedule.schedule_origin_generation",
    )
    explicit = schedule["explicit_next_generations"]
    if type(explicit) is not list:
        _fail("source.schedule.explicit_next_generations", "must be a list")
    explicit_values = [
        _strict_int(
            value,
            path=f"source.schedule.explicit_next_generations[{index}]",
            minimum=1,
        )
        for index, value in enumerate(explicit)
    ]
    if explicit_values != sorted(set(explicit_values)):
        _fail(
            "source.schedule.explicit_next_generations",
            "must be strictly increasing and unique",
        )
    if any(value <= origin for value in explicit_values):
        _fail(
            "source.schedule.explicit_next_generations",
            "must contain future coordinates after the schedule origin",
        )
    cancellation_due = _strict_bool(
        schedule["cancellation_due"], path="source.schedule.cancellation_due"
    )
    if cancellation_due != (terminal_status == "cancelled"):
        _fail(
            "source.schedule.cancellation_due",
            "must exactly reflect settled terminal cancellation",
        )
    coordinate = next_generation - origin
    return (
        (coordinate > 0 and coordinate % 8 == 0)
        or next_generation in explicit_values
        or cancellation_due
    )


def _null_paths(value: Any, *, path: str = "") -> list[str]:
    if value is None:
        return [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            if not path and key == "undefined_fields":
                continue
            child_path = f"{path}.{key}" if path else key
            paths.extend(_null_paths(child, path=child_path))
        return paths
    if type(value) is list:
        paths = []
        for index, child in enumerate(value):
            paths.extend(_null_paths(child, path=f"{path}[{index}]"))
        return paths
    return []


def _coerce_source(
    source: CompactSummarySourceEnvelope | Mapping[str, Any],
) -> CompactSummarySourceEnvelope:
    if isinstance(source, CompactSummarySourceEnvelope):
        return source
    if not isinstance(source, Mapping):
        _fail("source", "must be CompactSummarySourceEnvelope or a mapping")
    return CompactSummarySourceEnvelope.from_mapping(source)


def build_compact_summary_row(
    source: CompactSummarySourceEnvelope | Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate one canonical CPU row without mutating its sources."""

    envelope = _coerce_source(source)
    if envelope.schema_version != SOURCE_SCHEMA_VERSION:
        _fail(
            "source.schema_version",
            f"must equal {SOURCE_SCHEMA_VERSION!r}",
        )
    pre = _require_exact_mapping(
        envelope.pre_turnover,
        _PRE_TURNOVER_FIELDS,
        path="source.pre_turnover",
    )
    post = _require_exact_mapping(
        envelope.post_turnover,
        _POST_TURNOVER_FIELDS,
        path="source.post_turnover",
    )
    settled = _require_exact_mapping(
        envelope.settled, _SETTLED_FIELDS, path="source.settled"
    )
    terminal_source = _require_exact_mapping(
        settled["terminal"],
        _TERMINAL_AUTHORITY_FIELDS,
        path="source.settled.terminal",
    )
    digests = _require_exact_mapping(
        envelope.digests, _DIGEST_FIELDS, path="source.digests"
    )

    generation_start = _copy_exact_object(
        pre["generation_start"],
        ("generation", "tick"),
        path="source.pre_turnover.generation_start",
    )
    generation_end = _copy_exact_object(
        pre["generation_end"],
        ("generation", "tick_exclusive"),
        path="source.pre_turnover.generation_end",
    )
    next_generation_start = _copy_exact_object(
        post["next_generation_start"],
        ("generation", "tick"),
        path="source.post_turnover.next_generation_start",
    )
    generation = _strict_int(
        generation_end["generation"],
        path="source.pre_turnover.generation_end.generation",
    )
    next_generation = _strict_int(
        next_generation_start["generation"],
        path="source.post_turnover.next_generation_start.generation",
        minimum=1,
    )

    terminal = {
        "status": terminal_source["status"],
        "recoverable_error_status": terminal_source["recoverable_error_status"],
        "recoverable_error_code": terminal_source["recoverable_error_code"],
        "recoverable_error_digest_sha256": _recoverable_error_digest(
            terminal_source
        ),
        "sticky_failure": terminal_source["sticky_failure"],
        "failure_generation": terminal_source["failure_generation"],
        "failure_tick": terminal_source["failure_tick"],
        "failure_stage": terminal_source["failure_stage"],
    }
    world = pre["world"]
    if world == "flat":
        if envelope.patch is not None:
            _fail("source.patch", "must be null for Flat")
        patch_metrics = None
    elif world == "patch":
        patch_metrics = _copy_exact_object(
            envelope.patch,
            _PATCH_METRIC_FIELDS,
            path="source.patch",
        )
    else:
        _fail("source.pre_turnover.world", "must be 'flat' or 'patch'")

    row: dict[str, Any] = {
        "schema_version": WIRE_SCHEMA_VERSION,
        "summary_valid": settled["summary_valid"],
        "summary_status": settled["summary_status"],
        "world": world,
        "generation_start": generation_start,
        "generation_end": generation_end,
        "next_generation_start": next_generation_start,
        "alive": pre["alive"],
        "births": pre["births"],
        "failed_births": pre["failed_births"],
        "deaths": pre["deaths"],
        "deaths_by_class": _copy_exact_object(
            pre["deaths_by_class"],
            ("energy_depleted", "health_depleted"),
            path="source.pre_turnover.deaths_by_class",
        ),
        "turnover": _copy_exact_object(
            post["turnover"],
            ("count", "status"),
            path="source.post_turnover.turnover",
        ),
        "events": {
            "cursor_start": pre["event_cursor_start"],
            "cursor_pre_turnover": pre["event_cursor_pre_turnover"],
            "cursor_end": post["event_cursor_end"],
            "authoritative_cursor": post["authoritative_event_cursor"],
            "event_row_count": post["event_row_count"],
            "generation_event_count": pre["generation_event_count"],
            "turnover_event_count": post["turnover_event_count"],
            "generation_counts_by_type": _copy_exact_object(
                pre["generation_counts_by_type"],
                _EVENT_COUNT_FIELDS,
                path="source.pre_turnover.generation_counts_by_type",
            ),
            "turnover_counts_by_type": _copy_exact_object(
                post["turnover_counts_by_type"],
                _EVENT_COUNT_FIELDS,
                path="source.post_turnover.turnover_counts_by_type",
            ),
            "active_prefix_digest_sha256": _strict_sha256(
                digests["events_active_prefix_digest_sha256"],
                path="source.digests.events_active_prefix_digest_sha256",
            ),
        },
        "identity": _identity(post["identity"], post["active_agent_ids"]),
        "rng": _rng(envelope.rng),
        "terminal": terminal,
        "checkpoint_due": _checkpoint_due(
            envelope.schedule,
            next_generation=next_generation,
            terminal_status=terminal["status"],
        ),
        "state_digests": {
            field: _strict_sha256(
                digests[field], path=f"source.digests.{field}"
            )
            for field in _DIGEST_FIELDS
            if field != "events_active_prefix_digest_sha256"
        },
        "generation_metrics": _copy_exact_object(
            pre["generation_metrics"],
            _GENERATION_METRIC_FIELDS,
            path="source.pre_turnover.generation_metrics",
        ),
        "behavior_metrics": _behavior_metrics(envelope.behavior),
        "lineage_metrics": _lineage_metrics(
            envelope.lineage, generation=generation
        ),
        "patch_metrics": patch_metrics,
    }
    row["undefined_fields"] = sorted(_null_paths(row))
    validate_compact_summary_row(row)
    return row


def validate_compact_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row through the frozen wire validator without mutation."""

    if not isinstance(summary, Mapping):
        _fail("summary", "must be a mapping")
    from tools.scripts.eiel_cuda_architecture_b_compact_summary_validator import (
        validate_summary,
    )

    return validate_summary(summary)


def compact_summary_canonical_json_bytes(summary: Mapping[str, Any]) -> bytes:
    """Validate and encode one row with the frozen single trailing LF."""

    validate_compact_summary_row(summary)
    return _canonical_json_without_lf(summary) + b"\n"


def build_compact_summary_json_bytes(
    source: CompactSummarySourceEnvelope | Mapping[str, Any],
) -> bytes:
    """Build, validate, and canonically encode one CPU summary row."""

    return compact_summary_canonical_json_bytes(build_compact_summary_row(source))


# Explicit aliases keep the public surface discoverable to callers using CPU or
# canonical-row terminology without introducing a second implementation.
build_compact_summary_cpu = build_compact_summary_row
build_canonical_row = build_compact_summary_row
canonical_json_bytes = compact_summary_canonical_json_bytes


__all__ = [
    "CompactSummarySourceEnvelope",
    "CompactSummarySourceError",
    "RECOVERABLE_ERROR_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "WIRE_SCHEMA_VERSION",
    "build_canonical_row",
    "build_compact_summary_cpu",
    "build_compact_summary_json_bytes",
    "build_compact_summary_row",
    "canonical_json_bytes",
    "compact_summary_canonical_json_bytes",
    "validate_compact_summary_row",
]
