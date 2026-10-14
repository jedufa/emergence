from __future__ import annotations

"""Deterministic, read-only analytics over frozen Scientific Chronicle artifacts."""

from collections import Counter, defaultdict
import csv
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.schema import canonical_json, canonical_payload, event_order_key, validate_event
from simulation.chronicle.storage import ARTIFACT_NAMES, atomic_write_text, read_json, read_jsonl, write_json


ANALYTICS_SCHEMA_VERSION = "eiel_scientific_chronicle_analytics_v1"
GROUP_SCHEMA_VERSION = "eiel_scientific_chronicle_selected_run_summary_v1"
SEARCH_SCHEMA_VERSION = "eiel_scientific_chronicle_search_v1"
ALLOWED_SELECTION_MODES = frozenset({"explicit_operator_selection", "declared_group"})
SEVERITY_RANK = {"informational": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4}
MAX_SEARCH_RESULTS = 10_000


class ChronicleAnalyticsError(ValueError):
    """Raised when Chronicle analytics cannot preserve frozen input contracts."""


def _root(path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    nested = candidate / "chronicle"
    return nested if (nested / ARTIFACT_NAMES["events"]).is_file() else candidate


def _load_run(path: Path | str) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _root(path)
    events = read_jsonl(root / ARTIFACT_NAMES["events"])
    for event in events:
        validate_event(event)
    if events != sorted(events, key=event_order_key):
        raise ChronicleAnalyticsError("formal event feed is not in frozen total order")
    episodes_payload = read_json(root / ARTIFACT_NAMES["episodes"])
    if not isinstance(episodes_payload, list):
        raise ChronicleAnalyticsError("episodes.json must contain a list")
    episodes = [dict(value) for value in episodes_payload if isinstance(value, Mapping)]
    if len(episodes) != len(episodes_payload):
        raise ChronicleAnalyticsError("episodes.json contains a non-object record")
    for episode in episodes:
        if str(episode.get("schema_version")) != "eiel_scientific_episode_v1":
            raise ChronicleAnalyticsError("unsupported episode schema")
    contexts = read_jsonl(root / ARTIFACT_NAMES["contexts"])
    for context in contexts:
        if str(context.get("schema_version")) != "eiel_event_context_window_v1":
            raise ChronicleAnalyticsError("unsupported context-window schema")
    manifest = dict(read_json(root / ARTIFACT_NAMES["manifest"]))
    if str(manifest.get("schema_version")) != "eiel_scientific_chronicle_manifest_v1":
        raise ChronicleAnalyticsError("unsupported Chronicle manifest schema")
    summary = dict(read_json(root / ARTIFACT_NAMES["summary"]))
    storage = dict(read_json(root / ARTIFACT_NAMES["storage"]))
    event_ids = {str(event["event_id"]) for event in events}
    for episode in episodes:
        missing = set(str(value) for value in episode.get("source_event_ids", []) or []).difference(event_ids)
        if missing:
            raise ChronicleAnalyticsError(f"episode {episode.get('episode_id')} references missing formal events")
    return root, events, episodes, contexts, manifest, summary, storage


def _provenance(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(manifest.get("config", {}) or {})
    scenario = (
        manifest.get("scenario")
        or manifest.get("scenario_id")
        or config.get("scenario")
        or config.get("scenario_id")
        or "unknown"
    )
    seed = manifest.get("seed", config.get("seed"))
    return {
        "run_id": str(manifest.get("run_id") or summary.get("run_id") or "unknown"),
        "campaign_id": manifest.get("campaign_id", summary.get("campaign_id")),
        "branch_id": manifest.get("branch_id", summary.get("branch_id")),
        "parent_checkpoint_id": manifest.get("parent_checkpoint_id", summary.get("parent_checkpoint_id")),
        "source_commit": str(manifest.get("source_commit") or "unknown"),
        "scenario": str(scenario),
        "seed": seed if seed is not None else "unknown",
        "profile": str(manifest.get("profile") or summary.get("profile") or "unknown"),
        "source_completeness": str(manifest.get("source_completeness") or summary.get("source_completeness") or "unknown"),
        "finalized": bool(manifest.get("finalized", summary.get("finalized", False))),
    }


def _duration(episode: Mapping[str, Any]) -> int:
    return max(1, int(episode.get("end_generation", 0)) - int(episode.get("start_generation", 0)) + 1)


def _episode_overlap(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        episodes,
        key=lambda value: (
            int(value.get("start_generation", 0)),
            int(value.get("end_generation", 0)),
            str(value.get("episode_id", "")),
        ),
    )
    active: list[Mapping[str, Any]] = []
    overlap_pairs = 0
    family_pairs: Counter[str] = Counter()
    maximum_concurrent = 0
    for episode in ordered:
        start = int(episode.get("start_generation", 0))
        active = [value for value in active if int(value.get("end_generation", 0)) >= start]
        for prior in active:
            overlap_pairs += 1
            pair = tuple(sorted((str(prior.get("episode_type", "unknown")), str(episode.get("episode_type", "unknown")))))
            family_pairs[" + ".join(pair)] += 1
        active.append(episode)
        maximum_concurrent = max(maximum_concurrent, len(active))
    return {
        "overlapping_episode_pair_count": overlap_pairs,
        "maximum_concurrent_episodes": maximum_concurrent,
        "overlap_counts_by_family_pair": dict(sorted(family_pairs.items())),
    }


def _density(events: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    event_counts = Counter(int(event.get("generation", 0)) for event in events)
    episode_starts = Counter(int(episode.get("start_generation", 0)) for episode in episodes)
    episode_ends = Counter(int(episode.get("end_generation", 0)) for episode in episodes)
    deltas: Counter[int] = Counter()
    for episode in episodes:
        start = int(episode.get("start_generation", 0))
        end = int(episode.get("end_generation", 0))
        deltas[start] += 1
        deltas[end + 1] -= 1
    generations = sorted(
        set(event_counts)
        | set(episode_starts)
        | set(episode_ends)
        | set(deltas)
    )
    if not generations:
        return []
    active = 0
    rows: list[dict[str, Any]] = []
    for generation in range(min(generations), max(generations) + 1):
        active += deltas[generation]
        rows.append(
            {
                "generation": generation,
                "event_count": event_counts[generation],
                "episode_start_count": episode_starts[generation],
                "episode_end_count": episode_ends[generation],
                "active_episode_count": active,
            }
        )
    return rows


def _region_key(event: Mapping[str, Any]) -> str | None:
    region = event.get("region_id")
    if region:
        return str(region)
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return f"patch:{int(coordinates[0])},{int(coordinates[1])}"
    return None


def _mutation_links(root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    candidates = (
        root / "genome_attribution" / "mutation_events.jsonl",
        root.parent / "genome_attribution" / "mutation_events.jsonl",
    )
    path = next((value for value in candidates if value.is_file()), None)
    by_lineage: dict[str, list[str]] = defaultdict(list)
    by_agent: dict[str, list[str]] = defaultdict(list)
    if path is None:
        return {}, {}
    for row in read_jsonl(path):
        mutation_id = str(row.get("mutation_event_id") or "")
        lineage = str(row.get("lineage_id") or "")
        child = str(row.get("child_agent_id") or "")
        if mutation_id and lineage:
            by_lineage[lineage].append(mutation_id)
        if mutation_id and child:
            by_agent[child].append(mutation_id)
    return (
        {key: sorted(set(values)) for key, values in sorted(by_lineage.items())},
        {key: sorted(set(values)) for key, values in sorted(by_agent.items())},
    )


def _lineage_histories(
    root: Path,
    events: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mutation_by_lineage, _mutation_by_agent = _mutation_links(root)
    lineages = sorted(
        {str(value) for event in events for value in event.get("lineage_ids", []) or []}
        | {str(value) for episode in episodes for value in episode.get("lineage_ids", []) or []}
    )
    rows: list[dict[str, Any]] = []
    for lineage in lineages:
        lineage_events = [event for event in events if lineage in {str(value) for value in event.get("lineage_ids", []) or []}]
        lineage_episodes = [episode for episode in episodes if lineage in {str(value) for value in episode.get("lineage_ids", []) or []}]
        generations = [int(event.get("generation", 0)) for event in lineage_events]
        region_keys = sorted({value for event in lineage_events if (value := _region_key(event))})
        ecology_associations = sorted(
            {
                str(episode.get("episode_type", ""))
                for episode in lineage_episodes
                if str(episode.get("episode_type", ""))
                in {"regional_food_depletion", "crowding_waste_crisis", "predator_pressure_wave", "delayed_hazard_cycle", "ecological_recovery"}
            }
        )
        rows.append(
            {
                "lineage_id": lineage,
                "first_recorded_generation": min(generations) if generations else None,
                "last_recorded_generation": max(generations) if generations else None,
                "event_count": len(lineage_events),
                "event_counts_by_type": dict(sorted(Counter(str(event.get("event_type", "unknown")) for event in lineage_events).items())),
                "episode_count": len(lineage_episodes),
                "episode_counts_by_family": dict(sorted(Counter(str(episode.get("episode_type", "unknown")) for episode in lineage_episodes).items())),
                "expansion_episode_ids": sorted(str(episode["episode_id"]) for episode in lineage_episodes if str(episode.get("episode_type")) == "lineage_expansion"),
                "decline_episode_ids": sorted(str(episode["episode_id"]) for episode in lineage_episodes if str(episode.get("episode_type")) in {"lineage_decline", "lineage_extinction_sequence"}),
                "dominance_event_ids": sorted(str(event["event_id"]) for event in lineage_events if str(event.get("event_type")) == "lineage_dominance_crossing"),
                "near_extinction_event_ids": sorted(str(event["event_id"]) for event in lineage_events if str(event.get("event_type")) == "lineage_near_extinction"),
                "extinction_event_ids": sorted(str(event["event_id"]) for event in lineage_events if str(event.get("event_type")) == "lineage_extinction"),
                "branch_candidate_event_ids": sorted(str(event["event_id"]) for event in lineage_events if str(event.get("event_type")) == "branch_candidate"),
                "affected_regions": region_keys,
                "recorded_ecology_associations": ecology_associations,
                "mutation_event_ids": mutation_by_lineage.get(lineage, []),
                "causal_status": "not_assessed",
                "limitations": ["Co-occurrence with an episode is a recorded temporal association, not evidence that the lineage caused or was favored by it."],
            }
        )
    return rows


def _region_histories(events: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {value for event in events if (value := _region_key(event))}
        | {str(episode.get("affected_region")) for episode in episodes if episode.get("affected_region")}
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        region_events = [event for event in events if _region_key(event) == key]
        region_episodes = [episode for episode in episodes if str(episode.get("affected_region") or "") == key]
        generations = [int(event.get("generation", 0)) for event in region_events]
        lineages = sorted({str(value) for event in region_events for value in event.get("lineage_ids", []) or []})
        rows.append(
            {
                "region_id": key,
                "first_recorded_generation": min(generations) if generations else min((int(value.get("start_generation", 0)) for value in region_episodes), default=None),
                "last_recorded_generation": max(generations) if generations else max((int(value.get("end_generation", 0)) for value in region_episodes), default=None),
                "event_count": len(region_events),
                "event_counts_by_type": dict(sorted(Counter(str(event.get("event_type", "unknown")) for event in region_events).items())),
                "episode_count": len(region_episodes),
                "depletion_episode_ids": sorted(str(value["episode_id"]) for value in region_episodes if str(value.get("episode_type")) == "regional_food_depletion"),
                "crowding_waste_episode_ids": sorted(str(value["episode_id"]) for value in region_episodes if str(value.get("episode_type")) == "crowding_waste_crisis"),
                "predator_wave_episode_ids": sorted(str(value["episode_id"]) for value in region_episodes if str(value.get("episode_type")) == "predator_pressure_wave"),
                "recovery_episode_ids": sorted(str(value["episode_id"]) for value in region_episodes if str(value.get("episode_type")) == "ecological_recovery"),
                "delayed_hazard_episode_ids": sorted(str(value["episode_id"]) for value in region_episodes if str(value.get("episode_type")) == "delayed_hazard_cycle"),
                "lineage_ids": lineages,
                "population_event_ids": sorted(str(event["event_id"]) for event in region_events if str(event.get("event_category")) == "population"),
                "causal_status": "not_assessed",
            }
        )
    return rows


def analyze_run(path: Path | str) -> dict[str, Any]:
    root, events, episodes, contexts, manifest, summary, storage = _load_run(path)
    event_categories = Counter(str(event.get("event_category", "unknown")) for event in events)
    event_types = Counter(str(event.get("event_type", "unknown")) for event in events)
    severity_counts = Counter(str(event.get("severity", "unknown")) for event in events)
    priority_counts = Counter(str(event.get("live_priority", "unknown")) for event in events)
    episode_families = Counter(str(episode.get("episode_type", "unknown")) for episode in episodes)
    episode_severity = Counter(str(episode.get("peak_severity", "unknown")) for episode in episodes)
    durations = [_duration(episode) for episode in episodes]
    episodes_by_region = Counter(str(episode.get("affected_region") or "global") for episode in episodes)
    episodes_by_lineage: Counter[str] = Counter(
        str(lineage) for episode in episodes for lineage in episode.get("lineage_ids", []) or []
    )
    source_components = Counter(str(event.get("source_component", "unknown")) for event in events)
    confidence = Counter(str(event.get("confidence_class", "unknown")) for event in events)
    provenance = _provenance(manifest, summary)
    result = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "provenance": provenance,
        "source_root": str(root),
        "events": {
            "total": len(events),
            "by_category": dict(sorted(event_categories.items())),
            "by_type": dict(sorted(event_types.items())),
            "by_severity": dict(sorted(severity_counts.items())),
            "by_priority": dict(sorted(priority_counts.items())),
            "by_source_component": dict(sorted(source_components.items())),
        },
        "episodes": {
            "total": len(episodes),
            "by_family": dict(sorted(episode_families.items())),
            "durations_generations": durations,
            "duration_summary": {
                "minimum": min(durations) if durations else None,
                "maximum": max(durations) if durations else None,
                "mean": (sum(durations) / len(durations)) if durations else None,
            },
            "by_peak_severity": dict(sorted(episode_severity.items())),
            "by_region": dict(sorted(episodes_by_region.items())),
            "by_lineage": dict(sorted(episodes_by_lineage.items())),
            "overlap": _episode_overlap(episodes),
        },
        "density_by_generation": _density(events, episodes),
        "context_windows": {
            "count": len(contexts),
            "trigger_event_count": sum(len(context.get("trigger_event_ids", []) or []) for context in contexts),
        },
        "suppression": {
            "by_reason": dict(sorted(dict(summary.get("suppressed_counts", manifest.get("suppressed_counts", {})) or {}).items())),
            "total": int(storage.get("suppressed_count", 0) or 0),
        },
        "source_completeness": {
            "run_classification": provenance["source_completeness"],
            "event_counts_by_confidence": dict(sorted(confidence.items())),
            "source_component_counts": dict(sorted(source_components.items())),
        },
        "storage": {
            "total_bytes": int(storage.get("total_bytes", 0) or 0),
            "artifacts": list(storage.get("artifacts", []) or []),
        },
        "lineage_histories": _lineage_histories(root, events, episodes),
        "region_histories": _region_histories(events, episodes),
        "limitations": list(summary.get("limitations", []) or []) + [
            "Analytics summarize frozen Chronicle records and do not add causal, learning, benefit, strategy, or broad cognition claims."
        ],
    }
    return canonical_payload(result)


def _exact_match(event: Mapping[str, Any], episode: Mapping[str, Any] | None, query: Mapping[str, Any]) -> bool:
    exact_id = str(query.get("id") or "")
    ids = {
        str(event.get("event_id") or ""),
        str(event.get("subject_id") or ""),
        *(str(value) for value in event.get("source_record_refs", []) or []),
        *(str(value) for value in event.get("tags", []) or []),
    }
    if episode:
        ids.add(str(episode.get("episode_id") or ""))
    if exact_id and exact_id not in ids:
        return False
    agent_id = query.get("agent_id")
    if agent_id is not None and int(agent_id) not in {int(value) for value in event.get("agent_ids", []) or []}:
        return False
    lineage_id = str(query.get("lineage_id") or "")
    if lineage_id and lineage_id not in {str(value) for value in event.get("lineage_ids", []) or []}:
        return False
    region_id = str(query.get("region_id") or "")
    if region_id and region_id not in {str(event.get("region_id") or ""), str(_region_key(event) or "")}:
        return False
    tag = str(query.get("tag") or "")
    if tag and tag not in {str(value) for value in event.get("tags", []) or []}:
        return False
    event_type = str(query.get("event_type") or "")
    if event_type and str(event.get("event_type")) != event_type:
        return False
    family = str(query.get("episode_family") or "")
    if family and (episode is None or str(episode.get("episode_type")) != family):
        return False
    generation = int(event.get("generation", 0))
    if query.get("generation_start") is not None and generation < int(query["generation_start"]):
        return False
    if query.get("generation_end") is not None and generation > int(query["generation_end"]):
        return False
    importance = int(event.get("scientific_importance", 0))
    if query.get("importance_min") is not None and importance < int(query["importance_min"]):
        return False
    if query.get("importance_max") is not None and importance > int(query["importance_max"]):
        return False
    return True


def search_run(path: Path | str, query: Mapping[str, Any], *, limit: int = 1_000) -> dict[str, Any]:
    root, events, episodes, _contexts, manifest, summary, _storage = _load_run(path)
    maximum = max(1, min(MAX_SEARCH_RESULTS, int(limit)))
    entries = read_jsonl(root / ARTIFACT_NAMES["entries"])
    entries_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entry_target_ids: dict[str, set[str]] = {}
    for entry in entries:
        if str(entry.get("schema_version")) != "eiel_scientific_chronicle_entry_v1":
            raise ChronicleAnalyticsError("unsupported Chronicle entry schema")
        source_ids = {str(value) for value in entry.get("source_event_ids", []) or []}
        entry_target_ids[str(entry.get("chronicle_entry_id") or "")] = source_ids
        for event_id in source_ids:
            entries_by_event[event_id].append(entry)
    episodes_by_event: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        for event_id in episode.get("source_event_ids", []) or []:
            episodes_by_event.setdefault(str(event_id), episode)
    selected_events: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    exact_id = str(query.get("id") or "")
    target_event_ids = entry_target_ids.get(exact_id)
    event_query = dict(query)
    if target_event_ids is not None:
        event_query.pop("id", None)
    for event in events:
        episode = episodes_by_event.get(str(event["event_id"]))
        if target_event_ids is not None and str(event["event_id"]) not in target_event_ids:
            continue
        if _exact_match(event, episode, event_query):
            selected_events.append(event)
            if episode:
                episode_ids.add(str(episode["episode_id"]))
            if len(selected_events) >= maximum:
                break
    selected_episodes = [episode for episode in episodes if str(episode.get("episode_id")) in episode_ids]
    selected_event_ids = {str(event["event_id"]) for event in selected_events}
    selected_entries = [
        entry for entry in entries
        if selected_event_ids.intersection(str(value) for value in entry.get("source_event_ids", []) or [])
    ]
    mutation_paths = (
        root / "genome_attribution" / "mutation_events.jsonl",
        root.parent / "genome_attribution" / "mutation_events.jsonl",
    )
    mutation_path = next((value for value in mutation_paths if value.is_file()), None)
    mutation_events: list[dict[str, Any]] = []
    if mutation_path is not None:
        for mutation in read_jsonl(mutation_path):
            if exact_id and exact_id not in {
                str(mutation.get("mutation_event_id") or ""),
                str(mutation.get("genome_delta_id") or ""),
                *(str(value) for value in mutation.get("source_record_refs", []) or []),
            }:
                continue
            if query.get("agent_id") is not None and int(query["agent_id"]) not in {
                int(mutation.get("child_agent_id", -1)),
                *(int(value) for value in mutation.get("parent_agent_ids", []) or []),
            }:
                continue
            if query.get("lineage_id") and str(mutation.get("lineage_id") or "") != str(query["lineage_id"]):
                continue
            generation = int(mutation.get("generation", 0))
            if query.get("generation_start") is not None and generation < int(query["generation_start"]):
                continue
            if query.get("generation_end") is not None and generation > int(query["generation_end"]):
                continue
            mutation_events.append(mutation)
            if len(mutation_events) >= maximum:
                break
    return canonical_payload(
        {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "provenance": _provenance(manifest, summary),
            "query": dict(query),
            "limit": maximum,
            "event_count": len(selected_events),
            "events": selected_events,
            "entries": selected_entries,
            "episodes": selected_episodes,
            "mutation_events": mutation_events,
            "truncated": len(selected_events) == maximum,
            "limitations": ["Search is exact-field matching over frozen Chronicle records; it does not infer semantic similarity."],
        }
    )


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "values": ordered,
        "minimum": ordered[0] if ordered else None,
        "maximum": ordered[-1] if ordered else None,
        "mean": (sum(ordered) / len(ordered)) if ordered else None,
    }


def aggregate_selected_runs(paths: Sequence[Path | str], grouping: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(grouping.get("selection_mode") or "")
    group_id = str(grouping.get("group_id") or "")
    reason = str(grouping.get("selection_reason") or "")
    if mode not in ALLOWED_SELECTION_MODES or not group_id or not reason:
        raise ChronicleAnalyticsError(
            "cross-run analytics require selection_mode, group_id, and selection_reason; silent pooling is forbidden"
        )
    if len(paths) < 2:
        raise ChronicleAnalyticsError("cross-run analytics require at least two explicitly selected runs")
    analyses = [analyze_run(path) for path in paths]
    run_ids = [str(value["provenance"]["run_id"]) for value in analyses]
    if len(set(run_ids)) != len(run_ids):
        raise ChronicleAnalyticsError("selected runs must have unique run IDs")
    pooling_keys = [str(value) for value in grouping.get("pooling_keys", []) or []]
    pooling_reasons: list[str] = []
    if mode == "explicit_operator_selection" and not pooling_keys:
        pooling_reasons.append("operator selected a descriptive run set without declaring pooling keys")
    if mode == "declared_group" and not pooling_keys:
        pooling_reasons.append("declared groups require explicit pooling_keys")
    for key in pooling_keys:
        values = {canonical_json(analysis["provenance"].get(key)) for analysis in analyses}
        if len(values) != 1:
            pooling_reasons.append(f"provenance field {key} differs across selected runs")
    families = sorted({family for analysis in analyses for family in analysis["episodes"]["by_family"]})
    family_occurrence = []
    timing: dict[str, list[int]] = defaultdict(list)
    durations: dict[str, list[int]] = defaultdict(list)
    severities: dict[str, Counter[str]] = defaultdict(Counter)
    sequences: Counter[str] = Counter()
    for path, analysis in zip(paths, analyses):
        _root_value, _events, episodes, _contexts, _manifest, _summary, _storage = _load_run(path)
        ordered = sorted(
            episodes,
            key=lambda value: (
                int(value.get("start_generation", 0)),
                int(value.get("start_step", 0)),
                str(value.get("episode_type", "")),
                str(value.get("episode_id", "")),
            ),
        )
        family_sequence = [str(value.get("episode_type", "unknown")) for value in ordered]
        for width in (2, 3):
            for index in range(max(0, len(family_sequence) - width + 1)):
                sequences[" -> ".join(family_sequence[index : index + width])] += 1
        for episode in ordered:
            family = str(episode.get("episode_type", "unknown"))
            timing[family].append(int(episode.get("start_generation", 0)))
            durations[family].append(_duration(episode))
            severities[family][str(episode.get("peak_severity", "unknown"))] += 1
    for family in families:
        present = [analysis for analysis in analyses if int(analysis["episodes"]["by_family"].get(family, 0)) > 0]
        family_occurrence.append(
            {
                "episode_family": family,
                "run_count": len(present),
                "selected_run_count": len(analyses),
                "occurrence_fraction": len(present) / len(analyses),
                "source_run_ids": sorted(str(value["provenance"]["run_id"]) for value in present),
            }
        )
    scenario_seed = [
        {
            "run_id": analysis["provenance"]["run_id"],
            "scenario": analysis["provenance"]["scenario"],
            "seed": analysis["provenance"]["seed"],
        }
        for analysis in analyses
    ]
    return canonical_payload(
        {
            "schema_version": GROUP_SCHEMA_VERSION,
            "grouping": {
                "selection_mode": mode,
                "group_id": group_id,
                "selection_reason": reason,
                "pooling_keys": pooling_keys,
                "pooling_eligible": not pooling_reasons,
                "pooling_blockers": pooling_reasons,
            },
            "selected_run_count": len(analyses),
            "source_runs": [analysis["provenance"] for analysis in analyses],
            "matched_scenario_seed_structure": scenario_seed,
            "episode_family_occurrence": family_occurrence,
            "episode_timing_distributions": {key: _distribution(values) for key, values in sorted(timing.items())},
            "episode_duration_distributions": {key: _distribution(values) for key, values in sorted(durations.items())},
            "episode_severity_distributions": {key: dict(sorted(values.items())) for key, values in sorted(severities.items())},
            "recurring_episode_sequences": [
                {"sequence": key, "occurrences": count}
                for key, count in sorted(sequences.items(), key=lambda pair: (-pair[1], pair[0]))
            ],
            "source_completeness_by_run": {
                str(analysis["provenance"]["run_id"]): str(analysis["provenance"]["source_completeness"])
                for analysis in analyses
            },
            "limitations": [
                "This is a deterministic description of an explicitly selected run set; pooling eligibility is reported separately.",
                "Occurrence, timing, persistence, enrichment, and outcome differences do not establish causation, learning, benefit, or strategy.",
            ],
        }
    )


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return output.getvalue()


def _markdown_run(result: Mapping[str, Any]) -> str:
    provenance = result["provenance"]
    lines = [
        "# Scientific Chronicle Analytics",
        "",
        f"- Run: `{provenance['run_id']}`",
        f"- Source commit: `{provenance['source_commit']}`",
        f"- Scenario: `{provenance['scenario']}`",
        f"- Seed: `{provenance['seed']}`",
        f"- Source completeness: `{provenance['source_completeness']}`",
        f"- Formal events: {result['events']['total']}",
        f"- Episodes: {result['episodes']['total']}",
        "",
        "## Episode Families",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in result["episodes"]["by_family"].items())
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in result.get("limitations", []))
    return "\n".join(lines) + "\n"


def _markdown_group(result: Mapping[str, Any]) -> str:
    grouping = result["grouping"]
    lines = [
        "# Scientific Chronicle Selected-Run Summary",
        "",
        f"- Group: `{grouping['group_id']}`",
        f"- Selection mode: `{grouping['selection_mode']}`",
        f"- Selected runs: {result['selected_run_count']}",
        f"- Pooling eligible: `{str(grouping['pooling_eligible']).lower()}`",
        "",
        "## Episode Family Occurrence",
        "",
    ]
    for row in result["episode_family_occurrence"]:
        lines.append(
            f"- `{row['episode_family']}` occurred in {row['run_count']} of {row['selected_run_count']} selected runs "
            f"({row['occurrence_fraction']:.6g})."
        )
    if grouping["pooling_blockers"]:
        lines.extend(["", "## Pooling Blockers", ""])
        lines.extend(f"- {value}" for value in grouping["pooling_blockers"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in result.get("limitations", []))
    return "\n".join(lines) + "\n"


def write_run_exports(path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    result = analyze_run(path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "chronicle_analytics.json", result)
    event_rows = [
        {"dimension": dimension, "value": key, "count": value}
        for dimension, values in (
            ("category", result["events"]["by_category"]),
            ("event_type", result["events"]["by_type"]),
            ("severity", result["events"]["by_severity"]),
            ("priority", result["events"]["by_priority"]),
        )
        for key, value in values.items()
    ]
    atomic_write_text(output / "event_counts.csv", _csv_text(("dimension", "value", "count"), event_rows))
    episode_rows = [
        {"episode_family": key, "count": value}
        for key, value in result["episodes"]["by_family"].items()
    ]
    atomic_write_text(output / "episode_counts.csv", _csv_text(("episode_family", "count"), episode_rows))
    atomic_write_text(
        output / "density_by_generation.csv",
        _csv_text(("generation", "event_count", "episode_start_count", "episode_end_count", "active_episode_count"), result["density_by_generation"]),
    )
    atomic_write_text(output / "lineage_histories.csv", _csv_text(
        ("lineage_id", "first_recorded_generation", "last_recorded_generation", "event_count", "episode_count", "causal_status"),
        result["lineage_histories"],
    ))
    atomic_write_text(output / "region_histories.csv", _csv_text(
        ("region_id", "first_recorded_generation", "last_recorded_generation", "event_count", "episode_count", "causal_status"),
        result["region_histories"],
    ))
    atomic_write_text(output / "chronicle_analytics_report.md", _markdown_run(result))
    return canonical_payload({"output_dir": str(output), "run_id": result["provenance"]["run_id"], "files": sorted(path.name for path in output.iterdir() if path.is_file())})


def write_group_exports(paths: Sequence[Path | str], grouping: Mapping[str, Any], output_dir: Path | str) -> dict[str, Any]:
    result = aggregate_selected_runs(paths, grouping)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selected_run_summary.json", result)
    atomic_write_text(
        output / "episode_family_occurrence.csv",
        _csv_text(("episode_family", "run_count", "selected_run_count", "occurrence_fraction", "source_run_ids"), result["episode_family_occurrence"]),
    )
    atomic_write_text(output / "selected_run_summary.md", _markdown_group(result))
    return canonical_payload({"output_dir": str(output), "group_id": result["grouping"]["group_id"], "files": sorted(path.name for path in output.iterdir() if path.is_file())})
