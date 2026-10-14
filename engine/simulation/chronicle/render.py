from __future__ import annotations

"""Deterministic scientific rendering of formal events."""

from typing import Any, Iterable, Mapping

from simulation.chronicle.schema import CHRONICLE_ENTRY_SCHEMA_VERSION, canonical_payload, stable_hash


HEADLINES = {
    "generation_summary": "Generation {generation} summary",
    "notable_birth": "Notable birth recorded",
    "notable_death": "Notable death recorded",
    "birth_observation": "Birth recorded",
    "death_observation": "Death recorded",
    "birth_failure_observation": "Birth failure recorded",
    "mutation_observation": "Mutation record projected from native lifecycle data",
    "lineage_appearance": "Lineage {subject_id} appeared",
    "lineage_dominance_crossing": "Lineage {subject_id} crossed the dominance threshold",
    "lineage_leadership_change": "Lineage leadership changed",
    "lineage_rapid_share_increase": "Lineage {subject_id} increased rapidly",
    "lineage_rapid_share_decline": "Lineage {subject_id} declined rapidly",
    "lineage_near_extinction": "Lineage {subject_id} approached extinction",
    "lineage_extinction": "Lineage {subject_id} was no longer represented",
    "branch_candidate": "A reported lineage branch candidate met configured criteria",
    "regional_food_depletion": "Food depletion crossed the configured threshold",
    "regional_food_recovery": "Food depletion returned below the recovery threshold",
    "crowding_hotspot": "Crowding crossed the configured hotspot threshold",
    "waste_hotspot": "Waste crossed the configured hotspot threshold",
    "regrowth_suppression": "Food regrowth suppression was recorded",
    "predator_pressure_spike": "Predator pressure crossed the configured threshold",
    "predator_damage": "Predator damage was recorded",
    "ecological_recovery": "Positive ecological recovery activity was recorded",
    "population_bottleneck": "Population survival crossed the bottleneck threshold",
    "delayed_hazard_cycle": "A delayed-hazard cycle was recorded",
    "chronicle_integrity": "Chronicle integrity limitation recorded",
    "chronicle_events_suppressed": "Chronicle event candidates were suppressed by configured bounds",
    "chronicle_storage_budget": "Chronicle large-output storage profile was acknowledged",
    "checkpoint_chronicle_metadata_mismatch": "Checkpoint Chronicle metadata did not match",
    "retrospective_source_incomplete": "Retrospective Chronicle source coverage was incomplete",
}


def _measurements_text(measurements: Mapping[str, Any]) -> str:
    pairs: list[str] = []
    for key, value in sorted(measurements.items()):
        if isinstance(value, float):
            rendered = f"{value:.6g}"
        elif isinstance(value, (str, int, bool)) or value is None:
            rendered = str(value)
        else:
            continue
        pairs.append(f"{key}={rendered}")
    return ", ".join(pairs[:8])


def _value(measurements: Mapping[str, Any], name: str, default: str = "not available") -> str:
    value = measurements.get(name)
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _count(measurements: Mapping[str, Any], name: str, singular: str, plural: str | None = None) -> str:
    rendered = _value(measurements, name)
    noun = singular if rendered == "1" else (plural or f"{singular}s")
    return f"{rendered} {noun}"


def _lineage(event: Mapping[str, Any]) -> str:
    values = list(event.get("lineage_ids", []) or [])
    return str(values[0]) if values else str(event.get("subject_id") or "unidentified lineage")


def _location(event: Mapping[str, Any]) -> str:
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return f" at patch ({int(coordinates[0])}, {int(coordinates[1])})"
    region = event.get("region_id")
    return f" in region {region}" if region else ""


def render_scientific_summary(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type", "event"))
    generation = int(event.get("generation", 0))
    subject = str(event.get("subject_id") or "the recorded subject")
    measurements = dict(event.get("measurements", {}) or {})
    lineage = _lineage(event)
    location = _location(event)
    if event_type == "generation_summary":
        return (
            f"Generation {generation} ended with {_count(measurements, 'alive', 'living agent')}, "
            f"{_count(measurements, 'births', 'birth')}, {_count(measurements, 'deaths', 'death')}, "
            f"and {_count(measurements, 'failed_births', 'failed birth')}. "
            f"The Chronicle retained {_count(measurements, 'chronicle_event_count', 'event')} and "
            f"reported {_count(measurements, 'active_episode_count', 'active episode')}."
        )
    if event_type in {"notable_birth", "birth_observation"}:
        agent = str((event.get("agent_ids") or [subject])[0])
        reason = str(event.get("novelty_class") or "configured lifecycle criteria").replace("_", " ")
        return (
            f"Agent {agent} was recorded as born into lineage {lineage}{location}. "
            f"It was classified as {'notable' if event_type == 'notable_birth' else 'a full-profile observation'} "
            f"under the {reason} rule."
        )
    if event_type in {"notable_death", "death_observation"}:
        agent = str((event.get("agent_ids") or [subject])[0])
        return (
            f"Agent {agent} was recorded as dead in lineage {lineage}{location}; "
            f"the lineage living count after the event was {_value(measurements, 'lineage_alive_after')}."
        )
    if event_type == "lineage_appearance":
        return (
            f"Lineage {lineage} first appeared in the Chronicle at generation {generation} with "
            f"{_count(measurements, 'alive_count', 'living member')} and living-population share "
            f"{_value(measurements, 'alive_share')}."
        )
    if event_type == "lineage_dominance_crossing":
        return (
            f"Lineage {lineage} increased from share {_value(measurements, 'share_before')} to "
            f"{_value(measurements, 'share_after')}, crossing the configured dominance threshold "
            f"{_value(measurements, 'threshold')}."
        )
    if event_type == "lineage_leadership_change":
        before, separator, after = subject.partition("->")
        pair = f" from {before} to {after}" if separator else ""
        return (
            f"Living-population lineage leadership changed{pair} at generation {generation}; "
            f"the previous share was {_value(measurements, 'previous_leader_share')} and the new "
            f"leader share was {_value(measurements, 'new_leader_share')}."
        )
    if event_type in {"lineage_rapid_share_increase", "lineage_rapid_share_decline"}:
        direction = "increased" if event_type.endswith("increase") else "declined"
        return (
            f"Lineage {lineage} {direction} from share {_value(measurements, 'share_before')} to "
            f"{_value(measurements, 'share_after')}; the recorded share change was "
            f"{_value(measurements, 'share_delta')}."
        )
    if event_type == "lineage_near_extinction":
        return (
            f"Lineage {lineage} declined from {_count(measurements, 'alive_before', 'living member')} to "
            f"{_count(measurements, 'alive_after', 'living member')}, meeting the configured "
            "near-extinction condition."
        )
    if event_type == "lineage_extinction":
        return (
            f"Lineage {lineage} changed from {_count(measurements, 'previous_alive_count', 'living member')} "
            "to no living representatives at the generation boundary."
        )
    if event_type == "branch_candidate":
        return (
            f"The upstream lineage reporter identified {subject} as a branch candidate with "
            f"{_value(measurements, 'candidate_alive_count')} living members and share "
            f"{_value(measurements, 'candidate_alive_share')}; Chronicle records the report without "
            "inferring a new species or causal mechanism."
        )
    if event_type in {"regional_food_depletion", "regional_food_recovery"}:
        action = "rose above the depletion" if event_type.endswith("depletion") else "fell below the recovery"
        return (
            f"Food-depleted patch fraction {action} threshold at generation {generation}, changing "
            f"from {_value(measurements, 'depleted_fraction_before')} to "
            f"{_value(measurements, 'depleted_fraction_after')} against threshold "
            f"{_value(measurements, 'threshold')}."
        )
    if event_type == "crowding_hotspot":
        return (
            f"The crowding-hotspot fraction changed from {_value(measurements, 'hotspot_fraction_before')} "
            f"to {_value(measurements, 'hotspot_fraction_after')}, meeting threshold "
            f"{_value(measurements, 'threshold')}."
        )
    if event_type == "waste_hotspot":
        return (
            f"Mean waste on occupied patches changed from {_value(measurements, 'occupied_waste_before')} "
            f"to {_value(measurements, 'occupied_waste_after')}, meeting threshold "
            f"{_value(measurements, 'threshold')}."
        )
    if event_type == "regrowth_suppression":
        return (
            f"The environment recorded {_value(measurements, 'regrowth_suppressed_total')} total "
            "food-regrowth suppression during the generation."
        )
    if event_type == "predator_pressure_spike":
        return (
            f"The generation recorded {_value(measurements, 'predator_events')} predator events, "
            f"{_value(measurements, 'predator_damage')} total damage, and "
            f"{_value(measurements, 'predator_deaths')} predator-associated deaths; the configured "
            f"event threshold was {_value(measurements, 'threshold')}."
        )
    if event_type == "delayed_hazard_cycle":
        return (
            f"The delayed-hazard substrate recorded {_value(measurements, 'event_count')} events, "
            f"{_value(measurements, 'agents_damaged')} damaged agents, and "
            f"{_value(measurements, 'damage_total')} total damage during generation {generation}."
        )
    if event_type == "ecological_recovery":
        return (
            f"Positive ecological-recovery activity changed from "
            f"{_value(measurements, 'previous_total')} to "
            f"{_value(measurements, 'positive_recovery_total')} at generation {generation}."
        )
    if event_type == "population_bottleneck":
        return (
            f"{_value(measurements, 'alive')} of {_value(measurements, 'cohort_size')} cohort members "
            f"remained alive, giving survival fraction {_value(measurements, 'survival_fraction')} "
            f"against the configured bottleneck threshold {_value(measurements, 'threshold')}."
        )
    measured = _measurements_text(measurements)
    if str(event.get("event_category")) == "integrity":
        return f"Chronicle recorded the {event_type} integrity condition: {measured}."
    return f"At generation {generation}, Chronicle recorded {event_type}: {measured}."


def entry_from_event(event: Mapping[str, Any], *, episode_id: str | None = None) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_type = str(event.get("event_type", "event"))
    headline = HEADLINES.get(event_type, "Recorded scientific event").format(
        generation=int(event.get("generation", 0)),
        subject_id=str(event.get("subject_id") or "unknown"),
    )
    entry_id = "chr_" + stable_hash(
        "eiel-scientific-chronicle-entry-v1",
        {"event_id": event_id, "episode_id": episode_id},
    )
    limitations = ["This entry reports bounded recorded facts and does not infer agent mental states."]
    if str(event.get("causality_class")) not in {"direct_mechanism"}:
        limitations.append("Temporal ordering or association is not, by itself, evidence of causation.")
    return canonical_payload(
        {
            "schema_version": CHRONICLE_ENTRY_SCHEMA_VERSION,
            "chronicle_entry_id": entry_id,
            "run_id": str(event.get("run_id", "")),
            "campaign_id": event.get("campaign_id"),
            "branch_id": event.get("branch_id"),
            "start_generation": int(event.get("generation", 0)),
            "end_generation": int(event.get("generation", 0)),
            "start_step": int(event.get("step", 0)),
            "end_step": int(event.get("step", 0)),
            "entry_type": event_type,
            "headline": headline,
            "scientific_summary": render_scientific_summary(event),
            "key_measurements": dict(event.get("measurements", {}) or {}),
            "subject_references": {
                "subject_type": str(event.get("subject_type", "")),
                "subject_id": event.get("subject_id"),
                "agent_ids": list(event.get("agent_ids", []) or []),
                "lineage_ids": list(event.get("lineage_ids", []) or []),
                "region_id": event.get("region_id"),
                "patch_coordinates": event.get("patch_coordinates"),
            },
            "source_event_ids": [event_id],
            "episode_id": episode_id,
            "confidence_class": str(event.get("confidence_class", "unknown")),
            "causality_class": str(event.get("causality_class", "not_assessed")),
            "limitations": limitations,
            "tags": list(event.get("tags", []) or []),
            "importance": {
                "live_priority": str(event.get("live_priority", "background")),
                "scientific_importance": int(event.get("scientific_importance", 0)),
                "historical_importance": int(event.get("historical_importance", 0)),
                "public_interest": int(event.get("public_interest", 0)),
                "visual_interest": int(event.get("visual_interest", 0)),
            },
        }
    )


def entries_from_events(
    events: Iterable[Mapping[str, Any]],
    episodes: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    episode_by_event: dict[str, str] = {}
    for episode in episodes:
        for event_id in episode.get("source_event_ids", []) or []:
            episode_by_event.setdefault(str(event_id), str(episode["episode_id"]))
    return [entry_from_event(event, episode_id=episode_by_event.get(str(event["event_id"]))) for event in events]


def terminal_lines(entries: Iterable[Mapping[str, Any]], *, maximum: int) -> list[str]:
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "background": 4}
    selected = sorted(
        entries,
        key=lambda entry: (
            priority_order.get(str((entry.get("importance") or {}).get("live_priority", "background")), 4),
            int(entry.get("start_generation", 0)),
            str(entry.get("chronicle_entry_id", "")),
        ),
    )[: max(0, int(maximum))]
    return [
        f"[chronicle:{str((entry.get('importance') or {}).get('live_priority', 'background'))}] "
        f"{entry.get('headline')}: {entry.get('scientific_summary')}"
        for entry in selected
    ]
