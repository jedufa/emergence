from __future__ import annotations

"""Conservative deterministic episode detection over formal Chronicle events."""

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

from simulation.chronicle.schema import (
    EPISODE_SCHEMA_VERSION,
    canonical_payload,
    event_order_key,
    stable_hash,
)


EVENT_EPISODE_TYPES: dict[str, str] = {
    "regional_food_depletion": "regional_food_depletion",
    "regional_food_recovery": "ecological_recovery",
    "ecological_recovery": "ecological_recovery",
    "crowding_hotspot": "crowding_waste_crisis",
    "waste_hotspot": "crowding_waste_crisis",
    "regrowth_suppression": "crowding_waste_crisis",
    "predator_pressure_spike": "predator_pressure_wave",
    "predator_damage": "predator_pressure_wave",
    "lineage_appearance": "lineage_expansion",
    "lineage_dominance_crossing": "lineage_expansion",
    "lineage_rapid_share_increase": "lineage_expansion",
    "lineage_near_extinction": "lineage_decline",
    "lineage_rapid_share_decline": "lineage_decline",
    "lineage_extinction": "lineage_extinction_sequence",
    "population_bottleneck": "population_bottleneck",
    "delayed_hazard_cycle": "delayed_hazard_cycle",
}

INTERVAL_EPISODE_TYPES = frozenset(
    {
        "regional_food_depletion",
        "crowding_waste_crisis",
        "predator_pressure_wave",
        "ecological_recovery",
        "population_bottleneck",
        "delayed_hazard_cycle",
    }
)

INTERVAL_CLOSING_EVENT_TYPES: dict[str, frozenset[str]] = {
    "regional_food_depletion": frozenset({"regional_food_recovery"}),
}


def _episode_group(event: Mapping[str, Any], episode_type: str) -> str:
    if episode_type.startswith("lineage_"):
        lineages = list(event.get("lineage_ids", []) or [])
        return str(lineages[0] if lineages else event.get("subject_id") or "unknown_lineage")
    region_id = event.get("region_id")
    if region_id:
        return str(region_id)
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return f"patch:{int(coordinates[0])},{int(coordinates[1])}"
    return "global"


def _severity_rank(value: Any) -> int:
    return {
        "informational": 0,
        "minor": 1,
        "moderate": 2,
        "major": 3,
        "critical": 4,
    }.get(str(value), 0)


def _episode_record(
    *,
    run_id: str,
    branch_id: str | None,
    episode_type: str,
    group: str,
    events: list[Mapping[str, Any]],
    status: str,
    outcome: str,
) -> dict[str, Any]:
    first = events[0]
    last = events[-1]
    source_ids = [str(event["event_id"]) for event in events]
    lineage_ids = sorted({str(item) for event in events for item in event.get("lineage_ids", []) or []})
    agent_ids = sorted({int(item) for event in events for item in event.get("agent_ids", []) or []})
    peak = max(events, key=lambda event: (_severity_rank(event.get("severity")), event_order_key(event)))
    episode_id = "epi_" + stable_hash(
        "eiel-scientific-episode-v1",
        {
            "run_id": run_id,
            "branch_id": branch_id,
            "episode_type": episode_type,
            "group": group,
            "first_event_id": source_ids[0],
        },
    )
    return canonical_payload(
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "run_id": run_id,
            "branch_id": branch_id,
            "episode_type": episode_type,
            "status": status,
            "start_generation": int(first.get("generation", 0)),
            "start_step": int(first.get("step", 0)),
            "end_generation": int(last.get("generation", 0)),
            "end_step": int(last.get("step", 0)),
            "affected_region": None if group == "global" or episode_type.startswith("lineage_") else group,
            "agent_ids": agent_ids,
            "lineage_ids": lineage_ids,
            "source_event_ids": source_ids,
            "peak_severity": str(peak.get("severity", "informational")),
            "baseline_measurements": dict(first.get("measurements", {}) or {}),
            "final_measurements": dict(last.get("measurements", {}) or {}),
            "trigger_conditions": [str(first.get("event_type", ""))],
            "outcome": outcome,
            "confidence_class": "derived",
            "causality_class": "temporal_association",
            "limitations": [
                "Episode membership is based on fixed Chronicle V1 thresholds and temporal grouping.",
                "The episode does not establish causation beyond directly recorded mechanics.",
                *(
                    ["Interval boundaries use bounded generation-summary evidence from the formal event feed."]
                    if episode_type in INTERVAL_EPISODE_TYPES
                    else []
                ),
            ],
            "related_episode_ids": [],
        }
    )


def _measurement(event: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(dict(event.get("measurements", {}) or {}).get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _trigger_threshold(
    triggers: Iterable[Mapping[str, Any]],
    event_type: str,
    measurement: str,
    default: float,
) -> float:
    for event in triggers:
        if str(event.get("event_type")) == event_type:
            return _measurement(event, measurement, default)
    return float(default)


def _summary_active(
    episode_type: str,
    summary: Mapping[str, Any],
    triggers: list[Mapping[str, Any]],
) -> bool:
    if episode_type == "regional_food_depletion":
        recovery_threshold = _trigger_threshold(
            triggers,
            "regional_food_depletion",
            "recovery_threshold",
            0.10,
        )
        return _measurement(summary, "food_depleted_fraction") > recovery_threshold
    if episode_type == "crowding_waste_crisis":
        crowding_threshold = _trigger_threshold(triggers, "crowding_hotspot", "threshold", 0.20)
        waste_threshold = _trigger_threshold(triggers, "waste_hotspot", "threshold", 0.25)
        crowding_active = any(str(event.get("event_type")) == "crowding_hotspot" for event in triggers) and (
            _measurement(summary, "crowding_hotspot_fraction") >= crowding_threshold
        )
        waste_active = any(str(event.get("event_type")) == "waste_hotspot" for event in triggers) and (
            _measurement(summary, "occupied_waste_mean") >= waste_threshold
        )
        regrowth_active = any(str(event.get("event_type")) == "regrowth_suppression" for event in triggers) and (
            _measurement(summary, "regrowth_suppressed_total") > 0.0
        )
        return crowding_active or waste_active or regrowth_active
    if episode_type == "predator_pressure_wave":
        alive = max(1, int(_measurement(summary, "alive", 1.0)))
        threshold = max(1, int(math.ceil(alive * 0.05)))
        return _measurement(summary, "predator_events") >= threshold
    if episode_type == "ecological_recovery":
        return _measurement(summary, "recovery_positive_total") > 0.0
    if episode_type == "population_bottleneck":
        threshold = _trigger_threshold(triggers, "population_bottleneck", "threshold", 0.50)
        return _measurement(summary, "cohort_survival_fraction", 1.0) <= threshold
    if episode_type == "delayed_hazard_cycle":
        return _measurement(summary, "delayed_hazard_events") > 0.0
    return False


def _bounded_interval_episodes(
    *,
    run_id: str,
    branch_id: str | None,
    episode_type: str,
    group: str,
    triggers: list[Mapping[str, Any]],
    all_events: list[Mapping[str, Any]],
    finalize: bool,
) -> list[dict[str, Any]]:
    summaries = [
        event
        for event in all_events
        if str(event.get("run_id", "")) == run_id
        and (str(event.get("branch_id")) if event.get("branch_id") else None) == branch_id
        and str(event.get("event_type")) == "generation_summary"
    ]
    explicit_closers = [
        event
        for event in all_events
        if str(event.get("run_id", "")) == run_id
        and (str(event.get("branch_id")) if event.get("branch_id") else None) == branch_id
        and str(event.get("event_type")) in INTERVAL_CLOSING_EVENT_TYPES.get(episode_type, frozenset())
        and _episode_group(event, episode_type) == group
    ]
    episodes: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(triggers):
        start_generation = int(triggers[cursor].get("generation", 0))
        candidate_triggers = triggers[cursor:]
        start_triggers = [
            event
            for event in candidate_triggers
            if int(event.get("generation", 0)) == start_generation
        ]
        start_summary = next(
            (
                summary
                for summary in summaries
                if int(summary.get("generation", 0)) == start_generation
            ),
            None,
        )
        transition_only_recovery = bool(
            episode_type == "ecological_recovery"
            and start_summary is not None
            and any(str(event.get("event_type")) == "regional_food_recovery" for event in start_triggers)
            and not any(str(event.get("event_type")) == "ecological_recovery" for event in start_triggers)
            and not _summary_active(episode_type, start_summary, start_triggers)
        )
        if transition_only_recovery:
            episodes.append(
                _episode_record(
                    run_id=run_id,
                    branch_id=branch_id,
                    episode_type=episode_type,
                    group=group,
                    events=start_triggers,
                    status="closed",
                    outcome="recovery_transition_recorded",
                )
            )
            cursor += len(start_triggers)
            continue
        clear_summary: Mapping[str, Any] | None = None
        interval_summaries: list[Mapping[str, Any]] = []
        for summary in summaries:
            generation = int(summary.get("generation", 0))
            if generation < start_generation:
                continue
            interval_summaries.append(summary)
            if generation == start_generation:
                continue
            known_triggers = [
                event
                for event in candidate_triggers
                if int(event.get("generation", 0)) <= generation
            ]
            if _summary_active(episode_type, summary, known_triggers):
                continue
            clear_summary = summary
            break
        boundary_generation = (
            int(clear_summary.get("generation", 0)) if clear_summary is not None else None
        )
        if boundary_generation is None:
            segment_triggers = list(candidate_triggers)
            cursor = len(triggers)
        else:
            segment_triggers = [
                event
                for event in candidate_triggers
                if int(event.get("generation", 0)) <= boundary_generation
            ]
            cursor += max(1, len(segment_triggers))
        support: list[Mapping[str, Any]] = []
        if clear_summary is not None:
            support.extend(interval_summaries)
            support.extend(
                event
                for event in explicit_closers
                if int(event.get("generation", 0)) == boundary_generation
            )
            status = "closed"
            outcome = "condition_cleared"
        elif finalize:
            support.extend(interval_summaries)
            status = "closed"
            outcome = "run_end_finalization"
        else:
            support.extend(interval_summaries)
            status = "open"
            outcome = "provisional"
        unique: dict[str, Mapping[str, Any]] = {}
        for event in [*segment_triggers, *support]:
            unique[str(event["event_id"])] = event
        episode_events = sorted(unique.values(), key=event_order_key)
        episodes.append(
            _episode_record(
                run_id=run_id,
                branch_id=branch_id,
                episode_type=episode_type,
                group=group,
                events=episode_events,
                status=status,
                outcome=outcome,
            )
        )
    return episodes


def detect_episodes(
    events: Iterable[Mapping[str, Any]],
    *,
    gap_generations: int = 1,
    finalize: bool = False,
) -> list[dict[str, Any]]:
    """Group related events without adapting thresholds to the observed history."""
    all_events = sorted(events, key=event_order_key)
    relevant = [event for event in all_events if event.get("event_type") in EVENT_EPISODE_TYPES]
    groups: dict[tuple[str, str, str, str | None], list[Mapping[str, Any]]] = defaultdict(list)
    for event in relevant:
        episode_type = EVENT_EPISODE_TYPES[str(event["event_type"])]
        key = (
            str(event.get("run_id", "")),
            episode_type,
            _episode_group(event, episode_type),
            str(event.get("branch_id")) if event.get("branch_id") else None,
        )
        groups[key].append(event)

    episodes: list[dict[str, Any]] = []
    for (run_id, episode_type, group, branch_id), group_events in sorted(groups.items()):
        has_generation_summaries = any(
            str(event.get("run_id", "")) == run_id
            and (str(event.get("branch_id")) if event.get("branch_id") else None) == branch_id
            and str(event.get("event_type")) == "generation_summary"
            for event in all_events
        )
        if episode_type in INTERVAL_EPISODE_TYPES and group == "global" and has_generation_summaries:
            episodes.extend(
                _bounded_interval_episodes(
                    run_id=run_id,
                    branch_id=branch_id,
                    episode_type=episode_type,
                    group=group,
                    triggers=group_events,
                    all_events=all_events,
                    finalize=finalize,
                )
            )
            continue
        segment: list[Mapping[str, Any]] = []
        for event in group_events:
            if segment and int(event.get("generation", 0)) - int(segment[-1].get("generation", 0)) > int(
                gap_generations
            ):
                episodes.append(
                    _episode_record(
                        run_id=run_id,
                        branch_id=branch_id,
                        episode_type=episode_type,
                        group=group,
                        events=segment,
                        status="closed",
                        outcome="ended_after_event_gap",
                    )
                )
                segment = []
            segment.append(event)
            if str(event.get("event_type")) == "lineage_extinction":
                episodes.append(
                    _episode_record(
                        run_id=run_id,
                        branch_id=branch_id,
                        episode_type=episode_type,
                        group=group,
                        events=segment,
                        status="closed",
                        outcome="lineage_extinction_recorded",
                    )
                )
                segment = []
        if segment:
            episodes.append(
                _episode_record(
                    run_id=run_id,
                    branch_id=branch_id,
                    episode_type=episode_type,
                    group=group,
                    events=segment,
                    status="closed" if finalize else "open",
                    outcome="run_end_finalization" if finalize else "provisional",
                )
            )

    episodes.sort(
        key=lambda episode: (
            int(episode["start_generation"]),
            int(episode["start_step"]),
            str(episode["episode_type"]),
            str(episode["episode_id"]),
        )
    )
    for index, episode in enumerate(episodes):
        related: list[str] = []
        for other_index, other in enumerate(episodes):
            if index == other_index:
                continue
            overlap = not (
                int(other["end_generation"]) < int(episode["start_generation"])
                or int(other["start_generation"]) > int(episode["end_generation"])
            )
            if overlap:
                related.append(str(other["episode_id"]))
        episode["related_episode_ids"] = sorted(related)
    return canonical_payload(episodes)
