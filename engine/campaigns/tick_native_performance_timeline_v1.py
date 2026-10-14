"""Strict, receipt-only accounting for a tick-native launch.

The provider remains the authority for producing timestamps.  This module
only consumes a launch-bound envelope whose phase boundaries are already
recorded on one monotonic clock.  In particular, it never guesses GPU detach
from capture completion, or restore timing from a status flag.

The input is intentionally small::

    {
        "clock_domain": "perf_counter_ns",
        "wall_time": {"start_ns": 0, "end_ns": 1_000},
        "owner_preparation_receipts": [
            {"start_ns": 0, "end_ns": 100, "owner_id": "owner-0"},
        ],
        "simulation_receipts": [
            {"start_ns": 100, "end_ns": 300,
             "start_next_tick": 0, "end_next_tick": 1120},
        ],
        "checkpoint_receipts": [
            {"checkpoint_tick": 1120,
             "owner_span": {"start_next_tick": 0, "end_next_tick": 1120},
             "detach": {"start_ns": 300, "end_ns": 350},
             "persist": {"start_ns": 350, "end_ns": 900, "async": True},
             "owner_preparation": {"start_ns": 400, "end_ns": 500}},
        ],
        "administrative_receipts": [
            {"start_ns": 950, "end_ns": 1_000,
             "reason": "provider bookkeeping finalization"},
        ],
        "acs_tail_receipt": {"start_ns": 900, "end_ns": 950},
        "restore_receipt": {"status": "not_requested"},
    }

``start_perf_ns``/``end_perf_ns`` are accepted as an explicit spelling of a
monotonic pair for direct use with existing native receipts.  Unix timestamps
must not be mixed into phase arithmetic; the launch binder must normalize them
to the declared clock domain first.

``T_PERSIST`` is background service time.  It may overlap an exclusive phase
only when the receipt explicitly marks it ``async=True``.  Its reported
duration is the wall-covered persistence interval; service time is reported
separately.  The wall reconciliation uses the union of all classified
intervals, so asynchronous work is not double-counted.

Root ``owner_preparation_receipts`` are the independent owner-preparation
authority.  A checkpoint's ``owner_preparation`` is optional compatibility
metadata; any overlap with a root owner-preparation interval fails closed.
``administrative_receipts`` are optional, reason-bearing gap fillers for
provider bookkeeping.  They are reported separately and never enter a
primary phase duration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Final


TICK_NATIVE_PERFORMANCE_TIMELINE_SCHEMA_VERSION: Final[str] = (
    "eiel_tick_native_performance_timeline_v1"
)
RECONCILIATION_TOLERANCE_RATIO: Final[float] = 0.02
_PHASES: Final[tuple[str, ...]] = (
    "T_SIM",
    "T_DETACH",
    "T_PERSIST",
    "T_ACS_TAIL",
    "T_RESTORE",
    "OWNER_PREPARATION",
)
_BACKGROUND_PHASE = "T_PERSIST"
_ADMINISTRATIVE_PHASE = "ADMINISTRATIVE"


class TickNativePerformanceTimelineError(ValueError):
    """The launch-bound receipts cannot support strict phase accounting."""


def _fail(message: str) -> TickNativePerformanceTimelineError:
    return TickNativePerformanceTimelineError(message)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    return dict(value)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _fail(f"{field} must be an array")
    return list(value)


def _int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _pair(value: Mapping[str, Any], *, field: str) -> tuple[int, int]:
    """Read exactly one explicit monotonic timestamp pair."""

    pairs: list[tuple[int, int]] = []
    for start_name, end_name in (
        ("start_ns", "end_ns"),
        ("start_perf_ns", "end_perf_ns"),
    ):
        has_start = start_name in value
        has_end = end_name in value
        if has_start != has_end:
            raise _fail(f"{field} requires both {start_name} and {end_name}")
        if has_start:
            pairs.append(
                (
                    _int(value[start_name], field=f"{field}.{start_name}"),
                    _int(value[end_name], field=f"{field}.{end_name}"),
                )
            )
    if len(pairs) != 1:
        raise _fail(
            f"{field} requires one explicit start_ns/end_ns or "
            "start_perf_ns/end_perf_ns pair"
        )
    start, end = pairs[0]
    if end <= start:
        raise _fail(f"{field} timestamps must be strictly increasing")
    return start, end


def _phase(
    value: Any,
    *,
    phase: str,
    field: str,
    checkpoint_tick: int | None = None,
) -> dict[str, Any]:
    row = _mapping(value, field=field)
    start, end = _pair(row, field=field)
    asynchronous = row.get("async", False)
    if type(asynchronous) is not bool:
        raise _fail(f"{field}.async must be boolean")
    if phase != _BACKGROUND_PHASE and asynchronous:
        raise _fail(f"{field}.async is only valid for T_PERSIST")
    result: dict[str, Any] = {
        "phase": phase,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
        "asynchronous": asynchronous,
    }
    if checkpoint_tick is not None:
        result["checkpoint_tick"] = checkpoint_tick
    return result


def _checkpoint_tick(value: Mapping[str, Any], *, field: str) -> int:
    has_tick = "checkpoint_tick" in value
    has_coordinate = "coordinate" in value
    if not has_tick and not has_coordinate:
        raise _fail(f"{field} requires checkpoint_tick")
    tick = _int(
        value.get("checkpoint_tick", value.get("coordinate")),
        field=f"{field}.checkpoint_tick",
        minimum=0,
    )
    if has_tick and has_coordinate and value["checkpoint_tick"] != value["coordinate"]:
        raise _fail(f"{field} checkpoint_tick and coordinate disagree")
    return tick


def _owner_span(value: Any, *, field: str) -> dict[str, int] | None:
    if value is None:
        return None
    row = _mapping(value, field=field)
    start = _int(row.get("start_next_tick"), field=f"{field}.start_next_tick")
    end = _int(row.get("end_next_tick"), field=f"{field}.end_next_tick")
    if end <= start:
        raise _fail(f"{field} must be a positive half-open tick span")
    return {"start_next_tick": start, "end_next_tick": end}


def _public_interval(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": value["phase"],
        "start_ns": value["start_ns"],
        "end_ns": value["end_ns"],
        "duration_ns": value["duration_ns"],
        "asynchronous": value["asynchronous"],
        **(
            {
                "start_next_tick": value["start_next_tick"],
                "end_next_tick": value["end_next_tick"],
            }
            if "start_next_tick" in value
            else {}
        ),
        **(
            {"owner_span": dict(value["owner_span"])}
            if "owner_span" in value
            else {}
        ),
        **(
            {"owner_id": value["owner_id"]}
            if "owner_id" in value
            else {}
        ),
        **(
            {"source": value["source"]}
            if "source" in value
            else {}
        ),
        **(
            {"reason": value["reason"]}
            if "reason" in value
            else {}
        ),
        **(
            {"checkpoint_tick": value["checkpoint_tick"]}
            if "checkpoint_tick" in value
            else {}
        ),
    }


def _duration(intervals: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(item["duration_ns"]) for item in intervals)


def _union_duration(intervals: Sequence[Mapping[str, Any]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(
        ((int(item["start_ns"]), int(item["end_ns"])) for item in intervals),
        key=lambda pair: (pair[0], pair[1]),
    )
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _validate_order(intervals: Sequence[Mapping[str, Any]], *, phase: str) -> None:
    prior_start: int | None = None
    for index, item in enumerate(intervals):
        start = int(item["start_ns"])
        if prior_start is not None and start <= prior_start:
            raise _fail(
                f"{phase} timestamps are not strictly chronological at index {index}"
            )
        prior_start = start


def _validate_overlaps(
    intervals: Sequence[Mapping[str, Any]],
) -> None:
    for left, right in combinations(intervals, 2):
        overlap = min(int(left["end_ns"]), int(right["end_ns"])) - max(
            int(left["start_ns"]), int(right["start_ns"])
        )
        if overlap <= 0:
            continue
        left_async = left["phase"] == _BACKGROUND_PHASE and left["asynchronous"]
        right_async = right["phase"] == _BACKGROUND_PHASE and right["asynchronous"]
        if not (left_async or right_async):
            raise _fail(
                "phase intervals overlap without an explicit asynchronous "
                f"T_PERSIST classification: {left['phase']} and {right['phase']}"
            )
        if left["phase"] == _BACKGROUND_PHASE and not left["asynchronous"]:
            raise _fail("T_PERSIST overlap requires async=True")
        if right["phase"] == _BACKGROUND_PHASE and not right["asynchronous"]:
            raise _fail("T_PERSIST overlap requires async=True")


def _overlap_ns(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return min(int(left["end_ns"]), int(right["end_ns"])) - max(
        int(left["start_ns"]), int(right["start_ns"])
    )


def _validate_administrative_intervals(
    administrative: Sequence[Mapping[str, Any]],
    classified: Sequence[Mapping[str, Any]],
) -> None:
    prior: list[Mapping[str, Any]] = list(classified)
    for item in administrative:
        if any(_overlap_ns(item, other) > 0 for other in prior):
            raise _fail(
                "ADMINISTRATIVE interval overlaps an already classified phase; "
                "administrative time must fill an unclassified bookkeeping gap"
            )
        prior.append(item)


def _summary(
    phase: str, intervals: Sequence[Mapping[str, Any]], *, wall_covered: bool = False
) -> dict[str, Any]:
    service_ns = _duration(intervals)
    covered_ns = _union_duration(intervals)
    duration_ns = covered_ns if wall_covered else service_ns
    return {
        "intervals": [_public_interval(item) for item in intervals],
        "duration_ns": duration_ns,
        "seconds": duration_ns / 1_000_000_000.0,
        "service_duration_ns": service_ns,
        "service_seconds": service_ns / 1_000_000_000.0,
        "wall_covered_ns": covered_ns,
        "phase": phase,
    }


def derive_tick_native_performance_timeline_v1(
    launch_bound_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a strict phase timeline from one launch-bound receipt envelope.

    The caller must provide explicit same-clock phase endpoints.  Raw provider
    receipts that omit a boundary are rejected rather than approximated.
    """

    root = _mapping(launch_bound_receipt, field="launch_bound_receipt")
    clock_domain = root.get("clock_domain")
    if not isinstance(clock_domain, str) or not clock_domain:
        raise _fail("clock_domain is required")
    wall = _mapping(root.get("wall_time"), field="wall_time")
    wall_start, wall_end = _pair(wall, field="wall_time")
    wall_duration = wall_end - wall_start
    if wall_duration <= 0:
        raise _fail("wall_time duration must be positive")

    phase_rows: dict[str, list[dict[str, Any]]] = {phase: [] for phase in _PHASES}
    root_owner_receipts = _sequence(
        root.get("owner_preparation_receipts", []),
        field="owner_preparation_receipts",
    )
    root_owner_intervals: list[dict[str, Any]] = []
    for index, raw in enumerate(root_owner_receipts):
        field = f"owner_preparation_receipts[{index}]"
        row = _mapping(raw, field=field)
        interval = _phase(row, phase="OWNER_PREPARATION", field=field)
        interval["source"] = "root"
        if "owner_span" in row:
            owner_span = _owner_span(row.get("owner_span"), field=f"{field}.owner_span")
            if owner_span is not None:
                interval["owner_span"] = owner_span
        if "owner_id" in row:
            owner_id = row["owner_id"]
            if not isinstance(owner_id, str) or not owner_id:
                raise _fail(f"{field}.owner_id must be a non-empty string")
            interval["owner_id"] = owner_id
        root_owner_intervals.append(interval)
        phase_rows["OWNER_PREPARATION"].append(interval)

    simulations = _sequence(
        root.get("simulation_receipts"), field="simulation_receipts"
    )
    if not simulations:
        raise _fail("simulation_receipts must not be empty")
    for index, raw in enumerate(simulations):
        row = _mapping(raw, field=f"simulation_receipts[{index}]")
        start_tick = _int(
            row.get("start_next_tick"),
            field=f"simulation_receipts[{index}].start_next_tick",
        )
        end_tick = _int(
            row.get("end_next_tick"),
            field=f"simulation_receipts[{index}].end_next_tick",
        )
        if end_tick <= start_tick:
            raise _fail(
                f"simulation_receipts[{index}] must have a positive tick span"
            )
        interval = _phase(
            row,
            phase="T_SIM",
            field=f"simulation_receipts[{index}]",
        )
        interval["start_next_tick"] = start_tick
        interval["end_next_tick"] = end_tick
        phase_rows["T_SIM"].append(interval)

    checkpoints = _sequence(
        root.get("checkpoint_receipts"), field="checkpoint_receipts"
    )
    checkpoint_coordinates: list[dict[str, Any]] = []
    prior_checkpoint_tick: int | None = None
    for index, raw in enumerate(checkpoints):
        row = _mapping(raw, field=f"checkpoint_receipts[{index}]")
        field = f"checkpoint_receipts[{index}]"
        checkpoint_tick = _checkpoint_tick(row, field=field)
        if prior_checkpoint_tick is not None and checkpoint_tick <= prior_checkpoint_tick:
            raise _fail("checkpoint coordinates must be strictly increasing")
        prior_checkpoint_tick = checkpoint_tick
        detach = _phase(
            row.get("detach"),
            phase="T_DETACH",
            field=f"{field}.detach",
            checkpoint_tick=checkpoint_tick,
        )
        persist = _phase(
            row.get("persist"),
            phase="T_PERSIST",
            field=f"{field}.persist",
            checkpoint_tick=checkpoint_tick,
        )
        if persist["asynchronous"] is not True:
            # A synchronous persistence interval is still valid when it does
            # not overlap; requiring the explicit boolean prevents ambiguity.
            if "async" not in _mapping(row.get("persist"), field=f"{field}.persist"):
                raise _fail(f"{field}.persist.async must be explicit")
        owner_raw = row.get("owner_preparation")
        owner = (
            None
            if owner_raw is None
            else _phase(
                owner_raw,
                phase="OWNER_PREPARATION",
                field=f"{field}.owner_preparation",
                checkpoint_tick=checkpoint_tick,
            )
        )
        if owner is not None:
            owner["source"] = "checkpoint_compatibility"
        owner_span = _owner_span(row.get("owner_span"), field=f"{field}.owner_span")
        phase_rows["T_DETACH"].append(detach)
        phase_rows["T_PERSIST"].append(persist)
        if owner is not None:
            phase_rows["OWNER_PREPARATION"].append(owner)
        checkpoint_coordinates.append(
            {
                "checkpoint_tick": checkpoint_tick,
                "owner_span": owner_span,
                "T_DETACH": _public_interval(detach),
                "T_PERSIST": _public_interval(persist),
                "OWNER_PREPARATION": (
                    None if owner is None else _public_interval(owner)
                ),
            }
        )

    checkpoint_owner_intervals = [
        item
        for item in phase_rows["OWNER_PREPARATION"]
        if item.get("source") == "checkpoint_compatibility"
    ]
    for root_owner in root_owner_intervals:
        for checkpoint_owner in checkpoint_owner_intervals:
            if _overlap_ns(root_owner, checkpoint_owner) > 0:
                raise _fail(
                    "root and checkpoint compatibility owner-preparation "
                    "intervals duplicate or overlap"
                )

    tail_raw = root.get("acs_tail_receipt")
    tail = (
        None
        if tail_raw is None
        else _phase(tail_raw, phase="T_ACS_TAIL", field="acs_tail_receipt")
    )
    if tail is not None:
        phase_rows["T_ACS_TAIL"].append(tail)

    gpu_window_raw = root.get("gpu_performance_window")
    gpu_performance_window: dict[str, Any] | None = None
    if gpu_window_raw is not None:
        gpu_window_mapping = _mapping(
            gpu_window_raw, field="gpu_performance_window"
        )
        gpu_window_start, gpu_window_end = _pair(
            gpu_window_mapping, field="gpu_performance_window"
        )
        simulation_start = min(
            int(row["start_ns"]) for row in phase_rows["T_SIM"]
        )
        simulation_end = max(int(row["end_ns"]) for row in phase_rows["T_SIM"])
        if gpu_window_start != simulation_start:
            raise _fail(
                "gpu_performance_window must start at the first simulation dispatch"
            )
        if gpu_window_end < simulation_end:
            raise _fail(
                "gpu_performance_window ends before simulation completion"
            )
        if tail is None or gpu_window_end != int(tail["start_ns"]):
            raise _fail(
                "gpu_performance_window must end at the post-simulation ACS tail boundary"
            )
        if gpu_window_mapping.get("post_simulation_acs_tail_excluded") is not True:
            raise _fail(
                "gpu_performance_window must explicitly exclude the post-simulation ACS tail"
            )
        terminal_checkpoint_tick = _int(
            gpu_window_mapping.get("terminal_checkpoint_tick"),
            field="gpu_performance_window.terminal_checkpoint_tick",
            minimum=1,
        )
        expected_terminal_tick = max(
            int(row["end_next_tick"]) for row in phase_rows["T_SIM"]
        )
        if terminal_checkpoint_tick != expected_terminal_tick:
            raise _fail(
                "gpu_performance_window terminal checkpoint tick differs "
                "from simulation completion"
            )
        gpu_performance_window = {
            "start_ns": gpu_window_start,
            "end_ns": gpu_window_end,
            "duration_ns": gpu_window_end - gpu_window_start,
            "seconds": (gpu_window_end - gpu_window_start) / 1_000_000_000.0,
            "terminal_checkpoint_tick": terminal_checkpoint_tick,
            "post_simulation_acs_tail_excluded": True,
            "measurement_scope": (
                "first_simulation_dispatch_through_terminal_gpu_safe_release_v1"
            ),
        }

    restore = _mapping(root.get("restore_receipt"), field="restore_receipt")
    restore_status = restore.get("status")
    if restore_status == "not_requested":
        if any(
            key in restore
            for key in ("start_ns", "end_ns", "start_perf_ns", "end_perf_ns")
        ):
            raise _fail("not_requested restore must not contain timestamps")
        restore_interval = None
        restore_tick = None
    elif restore_status == "completed":
        restore_tick = _checkpoint_tick(restore, field="restore_receipt")
        restore_interval = _phase(
            restore, phase="T_RESTORE", field="restore_receipt", checkpoint_tick=restore_tick
        )
        phase_rows["T_RESTORE"].append(restore_interval)
    else:
        raise _fail("restore_receipt.status must be completed or not_requested")

    administrative_receipts = _sequence(
        root.get("administrative_receipts", []),
        field="administrative_receipts",
    )
    administrative_intervals: list[dict[str, Any]] = []
    for index, raw in enumerate(administrative_receipts):
        field = f"administrative_receipts[{index}]"
        row = _mapping(raw, field=field)
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise _fail(f"{field}.reason must be a non-empty string")
        interval = _phase(
            row,
            phase=_ADMINISTRATIVE_PHASE,
            field=field,
        )
        interval["reason"] = reason
        administrative_intervals.append(interval)

    for phase, rows in phase_rows.items():
        _validate_order(rows, phase=phase)
    _validate_order(administrative_intervals, phase=_ADMINISTRATIVE_PHASE)
    classified_intervals = [item for rows in phase_rows.values() for item in rows]
    for index, item in enumerate([*classified_intervals, *administrative_intervals]):
        if item["start_ns"] < wall_start or item["end_ns"] > wall_end:
            raise _fail(f"phase interval {index} lies outside wall_time")
    _validate_overlaps(classified_intervals)
    base_classified_union = _union_duration(classified_intervals)
    base_unclassified = wall_duration - base_classified_union
    if administrative_intervals and base_unclassified <= 0:
        raise _fail(
            "ADMINISTRATIVE intervals are not needed; no unclassified wall time remains"
        )
    _validate_administrative_intervals(
        administrative_intervals,
        classified_intervals,
    )
    all_intervals = [*classified_intervals, *administrative_intervals]

    classified_union = _union_duration(all_intervals)
    unclassified = wall_duration - classified_union
    if unclassified < 0:
        raise _fail("classified phase time exceeds wall_time")
    reconciliation_error_ratio = unclassified / wall_duration
    if reconciliation_error_ratio > RECONCILIATION_TOLERANCE_RATIO:
        raise _fail(
            "wall-time reconciliation error exceeds 2%: "
            f"{reconciliation_error_ratio:.6f}"
        )

    summaries = {
        phase: _summary(
            phase,
            phase_rows[phase],
            wall_covered=phase == _BACKGROUND_PHASE,
        )
        for phase in _PHASES
    }
    administrative_summary = _summary(
        _ADMINISTRATIVE_PHASE,
        administrative_intervals,
    )
    durations_ns = {
        "T_SIM": summaries["T_SIM"]["duration_ns"],
        "T_DETACH": summaries["T_DETACH"]["duration_ns"],
        "T_PERSIST": summaries["T_PERSIST"]["duration_ns"],
        "T_ACS_TAIL": summaries["T_ACS_TAIL"]["duration_ns"],
        "T_RESTORE": summaries["T_RESTORE"]["duration_ns"],
        "OWNER_PREPARATION": summaries["OWNER_PREPARATION"]["duration_ns"],
    }
    if gpu_performance_window is None:
        dispatch_overhead = None
    else:
        dispatch_domain_start = min(
            int(row["start_ns"]) for row in phase_rows["T_SIM"]
        )
        dispatch_domain_end = max(
            int(row["end_ns"]) for row in phase_rows["T_SIM"]
        )
        dispatch_domain_duration = dispatch_domain_end - dispatch_domain_start

        def clipped_phase_intervals(phase: str) -> list[dict[str, int]]:
            return [
                {
                    "start_ns": max(dispatch_domain_start, int(row["start_ns"])),
                    "end_ns": min(dispatch_domain_end, int(row["end_ns"])),
                }
                for row in phase_rows[phase]
                if int(row["end_ns"]) > dispatch_domain_start
                and int(row["start_ns"]) < dispatch_domain_end
            ]

        simulation_intervals = clipped_phase_intervals("T_SIM")
        simulation_union = _union_duration(simulation_intervals)
        non_dispatch_intervals = [
            interval
            for phase in ("T_DETACH", "T_PERSIST", "OWNER_PREPARATION")
            for interval in clipped_phase_intervals(phase)
        ]
        classified_union = _union_duration(
            [*simulation_intervals, *non_dispatch_intervals]
        )
        dispatch_gap = dispatch_domain_duration - classified_union
        if dispatch_gap < 0:
            raise _fail("simulation dispatch domain accounting is negative")
        dispatch_overhead = {
            "definition": "unclassified_simulation_dispatch_domain_fraction_v1",
            "dispatch_domain_start_ns": dispatch_domain_start,
            "dispatch_domain_end_ns": dispatch_domain_end,
            "dispatch_span_ns": dispatch_domain_duration,
            "simulation_dispatch_union_ns": simulation_union,
            "classified_non_dispatch_union_ns": classified_union - simulation_union,
            "capture_detach_union_ns": _union_duration(
                clipped_phase_intervals("T_DETACH")
            ),
            "persistence_union_ns": _union_duration(
                clipped_phase_intervals("T_PERSIST")
            ),
            "owner_preparation_union_ns": _union_duration(
                clipped_phase_intervals("OWNER_PREPARATION")
            ),
            "dispatch_gap_ns": dispatch_gap,
            "dispatch_domain_ns": simulation_union + dispatch_gap,
            "acs_tail_excluded": True,
            "fraction": dispatch_gap / (simulation_union + dispatch_gap),
        }
    return {
        "schema_version": TICK_NATIVE_PERFORMANCE_TIMELINE_SCHEMA_VERSION,
        "clock_domain": clock_domain,
        "durations_ns": durations_ns,
        "durations_seconds": {
            name: value / 1_000_000_000.0 for name, value in durations_ns.items()
        },
        "T_SIM_seconds": durations_ns["T_SIM"] / 1_000_000_000.0,
        "T_DETACH_seconds": durations_ns["T_DETACH"] / 1_000_000_000.0,
        "T_PERSIST_seconds": durations_ns["T_PERSIST"] / 1_000_000_000.0,
        "T_PERSIST_service_seconds": summaries["T_PERSIST"]["service_seconds"],
        "T_ACS_TAIL_seconds": durations_ns["T_ACS_TAIL"] / 1_000_000_000.0,
        "T_RESTORE_seconds": durations_ns["T_RESTORE"] / 1_000_000_000.0,
        "owner_preparation_seconds": (
            durations_ns["OWNER_PREPARATION"] / 1_000_000_000.0
        ),
        "phases": summaries,
        "owner_preparation": {
            "root_interval_count": len(root_owner_intervals),
            "checkpoint_compatibility_interval_count": len(
                checkpoint_owner_intervals
            ),
            "duplicate_check": "passed",
            "intervals": summaries["OWNER_PREPARATION"]["intervals"],
            "duration_ns": summaries["OWNER_PREPARATION"]["duration_ns"],
            "seconds": summaries["OWNER_PREPARATION"]["seconds"],
        },
        "administrative": {
            "intervals": administrative_summary["intervals"],
            "duration_ns": administrative_summary["duration_ns"],
            "seconds": administrative_summary["seconds"],
            "wall_covered_ns": administrative_summary["wall_covered_ns"],
            "present": bool(administrative_intervals),
            "excluded_from_primary_phase_durations": True,
            "reconciles_unclassified_wall_time": bool(administrative_intervals),
        },
        "checkpoint_coordinates": checkpoint_coordinates,
        "gpu_performance_window": gpu_performance_window,
        "dispatch_overhead": dispatch_overhead,
        "dispatch_overhead_fraction": (
            None if dispatch_overhead is None else dispatch_overhead["fraction"]
        ),
        "post_simulation_acs_tail": {
            "interval": (
                None if tail is None else _public_interval(tail)
            ),
            "duration_ns": summaries["T_ACS_TAIL"]["duration_ns"],
            "seconds": summaries["T_ACS_TAIL"]["seconds"],
            "excluded_from_gpu_performance_metrics": (
                gpu_performance_window is not None
            ),
            "metric_name": "T_ACS_TAIL",
        },
        "restore": {
            "status": restore_status,
            "checkpoint_tick": restore_tick,
            "interval": (
                None if restore_interval is None else _public_interval(restore_interval)
            ),
        },
        "wall_time_reconciliation": {
            "clock_domain": clock_domain,
            "start_ns": wall_start,
            "end_ns": wall_end,
            "wall_duration_ns": wall_duration,
            "primary_classified_union_ns": base_classified_union,
            "administrative_union_ns": administrative_summary["wall_covered_ns"],
            "classified_union_ns": classified_union,
            "unclassified_ns": unclassified,
            "reconciliation_error_ratio": reconciliation_error_ratio,
            "tolerance_ratio": RECONCILIATION_TOLERANCE_RATIO,
            "pass": True,
        },
        "strict": True,
        "async_persistence_overlap_explicit": True,
    }


account_tick_native_performance_timeline_v1 = (
    derive_tick_native_performance_timeline_v1
)
