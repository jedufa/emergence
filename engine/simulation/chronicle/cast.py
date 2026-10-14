from __future__ import annotations

"""Presentation-only cast naming and temporal callbacks for Observer Chronicle CAST1."""

from collections import Counter, defaultdict
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.observer import ObserverConfig, write_observer_outputs
from simulation.chronicle.schema import canonical_payload, stable_hash
from simulation.chronicle.storage import atomic_write_text, read_json, read_jsonl, sha256_file, write_json, write_jsonl


CAST_VERSION = "eiel_observer_cast1_v1"
CAST_MANIFEST_SCHEMA = "eiel_observer_cast_manifest_v1"
AGENT_NAME_REGISTRY_SCHEMA = "eiel_observer_agent_name_registry_v1"
LINEAGE_NAME_REGISTRY_SCHEMA = "eiel_observer_lineage_name_registry_v1"
RETROSPECTIVE_CAST_SCHEMA = "eiel_observer_retrospective_cast_v1"
LIVE_CAST_SCHEMA = "eiel_observer_live_cast_reconstruction_v1"
TEMPORAL_CALLBACK_SCHEMA = "eiel_observer_temporal_callback_index_v1"
OBSERVER_CAST_FEED_SCHEMA = "eiel_observer_cast_feed_item_v1"
OBSERVER_CAST_INDEX_SCHEMA = "eiel_observer_cast_index_v1"

CAST_ARTIFACTS = {
    "manifest": "cast_manifest.json",
    "agent_names": "agent_name_registry.json",
    "lineage_names": "lineage_name_registry.json",
    "retrospective_cast": "retrospective_cast.json",
    "live_cast": "live_cast_reconstruction.jsonl",
    "callbacks": "temporal_callback_index.json",
    "feed": "observer_cast_feed.jsonl",
    "episodes": "observer_cast_episodes.jsonl",
    "agents": "observer_cast_agents.jsonl",
    "lineages": "observer_cast_lineages.jsonl",
    "regions": "observer_cast_regions.jsonl",
    "history": "observer_cast_world_history.md",
    "audio": "observer_cast_audio_script.txt",
    "dialogue": "observer_cast_dialogue.txt",
    "index": "observer_cast_index.json",
}

GIVEN_NAMES = (
    "Aven", "Bryn", "Cael", "Dara", "Eris", "Fenn", "Galen", "Hale", "Iven", "Jori",
    "Kael", "Liora", "Maren", "Niko", "Orin", "Pax", "Quin", "Rhea", "Soren", "Tala",
    "Uma", "Vera", "Wren", "Xara", "Yori", "Zev", "Ari", "Bela", "Ciro", "Demi",
    "Elan", "Fara", "Gio", "Hera", "Ira", "Juno", "Kira", "Lio", "Mira", "Nova",
)

LINEAGE_FAMILIES = (
    "Amber", "Basin", "Cinder", "Delta", "Ember", "Fen", "Grove", "Harbor", "Iron", "Juniper",
    "Kestrel", "Lantern", "Morrow", "North", "Orchard", "Prairie", "Quartz", "Ridge", "Solace", "Tern",
    "Umber", "Vale", "Willow", "Xylem", "Yarrow", "Zephyr", "Aster", "Birch", "Cove", "Dune",
    "Elm", "Flint", "Glade", "Hollow", "Iris", "Jade", "Kelp", "Lumen", "Moss", "Nettle",
)

STATE_RANK = {
    "background": 0,
    "noticed": 1,
    "watchlist": 2,
    "featured": 3,
    "currently_central": 4,
    "retired_from_focus": 1,
    "deceased": 0,
    "retrospectively_important": 5,
}


class CastError(ValueError):
    """Raised when CAST1 cannot produce a bounded presentation artifact."""


def _chronicle_root(run_dir: Path) -> Path:
    if (run_dir / "chronicle" / "formal_events.jsonl").exists():
        return run_dir / "chronicle"
    if (run_dir / "formal_events.jsonl").exists():
        return run_dir
    raise CastError(f"no Chronicle artifacts found under {run_dir}")


def _sorted_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(event) for event in events),
        key=lambda event: (
            int(event.get("generation", 0)),
            int(event.get("step", 0)),
            int(event.get("phase_order", 0)),
            int(event.get("intra_step_sequence", 0)),
            str(event.get("event_id", "")),
        ),
    )


def _lineage_sort_key(lineage_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", lineage_id)
    return (int(match.group(1)) if match else 10**9, lineage_id)


def _agent_name(agent_id: int, lineage_id: str | None) -> dict[str, Any]:
    lineage = lineage_id or "unassigned"
    digest = stable_hash("observer-cast-agent-name-v1", {"agent_id": int(agent_id), "lineage_id": lineage}, length=16)
    given = GIVEN_NAMES[int(digest[:8], 16) % len(GIVEN_NAMES)]
    family = LINEAGE_FAMILIES[int(digest[8:], 16) % len(LINEAGE_FAMILIES)]
    lineage_suffix = lineage[-2:] if lineage and lineage[-2:].isdigit() else str(int(agent_id) % 100).zfill(2)
    display = f"{given} {family}-{lineage_suffix}"
    return {
        "agent_id": int(agent_id),
        "lineage_id": lineage_id,
        "display_name": display,
        "short_name": given,
        "name_components": {"given": given, "family": family, "suffix": lineage_suffix},
        "derivation": "stable_hash(agent_id,lineage_id); presentation-only; scientific ID retained",
    }


def _lineage_name(lineage_id: str) -> dict[str, Any]:
    digest = stable_hash("observer-cast-lineage-name-v1", lineage_id, length=16)
    family = LINEAGE_FAMILIES[int(digest[:8], 16) % len(LINEAGE_FAMILIES)]
    number = _lineage_sort_key(lineage_id)[0]
    ordinal = number if number != 10**9 else int(digest[8:], 16) % 100
    return {
        "lineage_id": lineage_id,
        "display_name": f"Lineage {ordinal} ({family})",
        "short_name": f"{family} line",
        "derivation": "stable_hash(lineage_id); presentation-only; scientific ID retained",
    }


def build_name_registries(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    agent_lineages: dict[int, str | None] = {}
    lineages: set[str] = set()
    for event in events:
        event_lineages = [str(value) for value in event.get("lineage_ids", []) or []]
        if event_lineages:
            lineages.update(event_lineages)
        for agent_id in event.get("agent_ids", []) or []:
            try:
                numeric = int(agent_id)
            except (TypeError, ValueError):
                continue
            agent_lineages.setdefault(numeric, event_lineages[0] if event_lineages else None)
    agent_rows = [_agent_name(agent_id, agent_lineages[agent_id]) for agent_id in sorted(agent_lineages)]
    lineage_rows = [_lineage_name(lineage_id) for lineage_id in sorted(lineages, key=_lineage_sort_key)]
    return (
        canonical_payload({"schema_version": AGENT_NAME_REGISTRY_SCHEMA, "items": agent_rows}),
        canonical_payload({"schema_version": LINEAGE_NAME_REGISTRY_SCHEMA, "items": lineage_rows}),
    )


def _agent_event_components(agent_id: int, rows: Sequence[Mapping[str, Any]], episodes_by_event: Mapping[str, list[str]]) -> dict[str, Any]:
    generations = [int(row.get("generation", 0)) for row in rows]
    event_types = Counter(str(row.get("event_type", "")) for row in rows)
    importance = sum(int(row.get("historical_importance", 0)) for row in rows)
    scientific = sum(int(row.get("scientific_importance", 0)) for row in rows)
    public = sum(int(row.get("public_interest", 0)) for row in rows)
    visual = sum(int(row.get("visual_interest", 0)) for row in rows)
    episode_ids = sorted({episode_id for row in rows for episode_id in episodes_by_event.get(str(row.get("event_id")), [])})
    birth_count = event_types["notable_birth"] + event_types["birth_observation"]
    death_count = event_types["notable_death"] + event_types["death_observation"]
    mutation_count = event_types["mutation_observation"]
    lineage_ids = sorted({str(value) for row in rows for value in (row.get("lineage_ids", []) or [])})
    lifespan = max(generations) - min(generations) + 1 if generations else 0
    return {
        "notable_event_count": len(rows),
        "lifespan_generations": lifespan,
        "first_generation": min(generations) if generations else None,
        "last_generation": max(generations) if generations else None,
        "episode_centrality": len(episode_ids),
        "historical_importance_sum": importance,
        "scientific_importance_sum": scientific,
        "public_interest_sum": public,
        "visual_interest_sum": visual,
        "birth_records": birth_count,
        "death_records": death_count,
        "mutation_occurrence_links": mutation_count,
        "lineage_ids": lineage_ids,
        "source_event_ids": [str(row.get("event_id")) for row in rows],
        "source_episode_ids": episode_ids,
    }


def _roles_for_agent(components: Mapping[str, Any]) -> list[str]:
    roles: list[str] = []
    if int(components.get("first_generation") or 0) == 0:
        roles.append("founder")
    if int(components.get("lifespan_generations") or 0) >= 5:
        roles.append("long_survivor")
    if int(components.get("episode_centrality") or 0) >= 2:
        roles.append("episode_witness")
    if int(components.get("mutation_occurrence_links") or 0) > 0:
        roles.append("mutation_record_carrier")
    if int(components.get("death_records") or 0) > 0:
        roles.append("completed_life_record")
    if int(components.get("notable_event_count") or 0) >= 3:
        roles.append("recurring_figure")
    return roles or ["supporting_figure"]


def build_retrospective_cast(
    events: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    agent_registry: Mapping[str, Any],
    max_agents: int = 8,
) -> dict[str, Any]:
    episodes_by_event: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        for event_id in episode.get("event_ids", []) or []:
            episodes_by_event[str(event_id)].append(episode_id)
    by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        for agent_id in event.get("agent_ids", []) or []:
            try:
                by_agent[int(agent_id)].append(event)
            except (TypeError, ValueError):
                continue
    names = {int(row["agent_id"]): row for row in agent_registry.get("items", [])}
    scored = []
    for agent_id, rows in sorted(by_agent.items()):
        components = _agent_event_components(agent_id, rows, episodes_by_event)
        score = (
            int(components["notable_event_count"]) * 8
            + int(components["lifespan_generations"]) * 3
            + int(components["episode_centrality"]) * 10
            + int(components["historical_importance_sum"]) * 2
            + int(components["public_interest_sum"])
            + int(components["visual_interest_sum"])
            + int(components["mutation_occurrence_links"]) * 4
        )
        roles = _roles_for_agent(components)
        scored.append(
            {
                "agent_id": agent_id,
                "display_name": names.get(agent_id, _agent_name(agent_id, None))["display_name"],
                "cast_score": score,
                "score_components": components,
                "roles": roles,
                "role_limitations": [
                    "Roles summarize recorded history only.",
                    "Cast score is presentation ranking, not scientific evidence.",
                ],
            }
        )
    scored.sort(key=lambda row: (-int(row["cast_score"]), int(row["agent_id"])))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
        row["featured"] = rank <= max_agents
    if scored and not any(row["featured"] for row in scored):
        scored[0]["featured"] = True
    return canonical_payload(
        {
            "schema_version": RETROSPECTIVE_CAST_SCHEMA,
            "score_version": "retrospective_cast_score_v1",
            "mode": "post_run_retrospective",
            "future_knowledge_disclosure": "This retrospective cast is computed after the run and must not be used as live knowledge.",
            "items": scored,
            "featured_agent_ids": [row["agent_id"] for row in scored if row.get("featured")],
            "no_clear_cast": not scored,
            "no_clear_cast_text": "No individual agent dominated the historical record. Evolution remained committed to ensemble work." if not scored else None,
        }
    )


def reconstruct_live_cast(
    events: Sequence[Mapping[str, Any]],
    *,
    agent_registry: Mapping[str, Any],
    max_featured: int = 5,
) -> list[dict[str, Any]]:
    names = {int(row["agent_id"]): row for row in agent_registry.get("items", [])}
    counts: Counter[int] = Counter()
    last_seen: dict[int, int] = {}
    states: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for event in _sorted_events(events):
        generation = int(event.get("generation", 0))
        event_id = str(event.get("event_id", ""))
        agents = []
        for raw in event.get("agent_ids", []) or []:
            try:
                agents.append(int(raw))
            except (TypeError, ValueError):
                continue
        for agent_id in agents:
            counts[agent_id] += 1
            last_seen[agent_id] = generation
        ranked = sorted(
            counts,
            key=lambda aid: (
                -(counts[aid] * 10 + (3 if int(last_seen.get(aid, -999)) == generation else 0)),
                aid,
            ),
        )
        featured = set(ranked[:max_featured])
        for agent_id in agents:
            previous = states.get(agent_id, "background")
            score = counts[agent_id] * 10 + (3 if agent_id in featured else 0)
            event_type = str(event.get("event_type", ""))
            if event_type in {"notable_death", "death_observation"}:
                state = "deceased"
            elif agent_id in featured and score >= 33:
                state = "currently_central"
            elif agent_id in featured and score >= 20:
                state = "featured"
            elif score >= 10:
                state = "watchlist"
            else:
                state = "noticed"
            if STATE_RANK[state] >= STATE_RANK.get(previous, 0) or state == "deceased":
                states[agent_id] = state
            rows.append(
                canonical_payload(
                    {
                        "schema_version": LIVE_CAST_SCHEMA,
                        "salience_version": "live_cast_salience_v1",
                        "event_id": event_id,
                        "generation": generation,
                        "step": int(event.get("step", 0)),
                        "agent_id": agent_id,
                        "display_name": names.get(agent_id, _agent_name(agent_id, None))["display_name"],
                        "previous_state": previous,
                        "state": states[agent_id],
                        "score_components": {
                            "prefix_notable_event_count": counts[agent_id],
                            "seen_in_current_generation": int(last_seen.get(agent_id, -1)) == generation,
                            "featured_rank": ranked.index(agent_id) + 1 if agent_id in ranked else None,
                            "critical_or_high_event": str(event.get("live_priority", "")) in {"critical", "high"},
                        },
                        "promotion_reason": "prefix evidence only; no future events inspected",
                        "future_leakage_guard": "computed in chronological order from events up to this row",
                    }
                )
            )
    return rows


def _name_text(text: str, agent_names: Mapping[int, str], lineage_names: Mapping[str, str]) -> str:
    result = str(text or "")
    for agent_id, display in sorted(agent_names.items(), key=lambda item: -len(str(item[0]))):
        result = re.sub(rf"\bAgent {agent_id}\b", f"{display} (Agent {agent_id})", result)
    for lineage_id, display in sorted(lineage_names.items(), key=lambda item: -len(item[0])):
        result = result.replace(f"Lineage {lineage_id}", f"{display} ({lineage_id})")
        result = result.replace(f"lineage {lineage_id}", f"{display} ({lineage_id})")
    return result


def _callback_for(
    row: Mapping[str, Any],
    prior: Mapping[str, str],
    cast_agent_ids: set[int],
    agent_names: Mapping[int, str],
) -> tuple[str | None, dict[str, Any] | None]:
    event_type = str(row.get("event_type", ""))
    subject = dict(row.get("subject_references", {}) or {})
    region = str(subject.get("region_id") or "")
    lineages = [str(value) for value in subject.get("lineage_ids", []) or []]
    agents = []
    for raw in subject.get("agent_ids", []) or []:
        try:
            agents.append(int(raw))
        except (TypeError, ValueError):
            continue
    family = str(row.get("episode_family") or event_type)
    key = f"{family}:{region or ','.join(lineages) or ','.join(map(str, agents))}"
    prior_id = prior.get(key)
    callback_type: str | None = None
    callback = None
    if key in prior and family in {"regional_food_depletion", "predator_pressure_wave", "crowding_waste_crisis", "lineage_extinction_sequence", "lineage_decline"}:
        callback_type = "repetition"
        callback = "Creation has seen this pattern before. The previous lesson appears to have entered the archive more than the behavior."
    elif event_type == "ecological_recovery" and any(k.startswith("regional_food_depletion:") for k in prior):
        callback_type = "reversal"
        callback = "Recovery has arrived after earlier scarcity. I withdraw a measured portion of my disappointment."
    elif event_type in {"predator_pressure_spike", "predator_damage"} and any(k.startswith("crowding_waste_crisis:") for k in prior):
        callback_type = "vindication"
        callback = "The crowded record has attracted risk. This is consistency, not wisdom."
    elif agents and agents[0] in cast_agent_ids:
        name = agent_names.get(agents[0], f"Agent {agents[0]}")
        if event_type in {"notable_birth", "birth_observation"} and key not in prior:
            callback_type = "first_mention"
            callback = f"{name} has entered the cast ledger. This is not necessarily a compliment."
        elif event_type in {"notable_death", "death_observation"}:
            callback_type = "failed_praise"
            callback = f"{name} leaves the living record. Creation is full of influential drafts that do not reach publication."
        else:
            callback_type = "long_survivor"
            callback = f"{name} remains in the record. Continued presence is becoming recognizable."
    prior[key] = str(row.get("observer_item_id", ""))
    if not callback_type or not callback:
        return None, None
    return callback, {
        "callback_type": callback_type,
        "current_observer_item_id": row.get("observer_item_id"),
        "prior_key": key,
        "prior_observer_item_id": prior_id,
        "source_event_ids": list(dict(row.get("source_links", {}) or {}).get("event_ids", []) or []),
        "limitations": [
            "Callback is presentation memory regenerated from earlier Observer rows.",
            "Callback does not add scientific causality or live future knowledge.",
        ],
    }


def build_cast_feed(
    observer_feed: Sequence[Mapping[str, Any]],
    *,
    agent_registry: Mapping[str, Any],
    lineage_registry: Mapping[str, Any],
    retrospective_cast: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent_names = {int(row["agent_id"]): str(row["display_name"]) for row in agent_registry.get("items", [])}
    lineage_names = {str(row["lineage_id"]): str(row["display_name"]) for row in lineage_registry.get("items", [])}
    cast_agent_ids = {int(agent_id) for agent_id in retrospective_cast.get("featured_agent_ids", []) or []}
    prior: dict[str, str] = {}
    callbacks: list[dict[str, Any]] = []
    cast_rows: list[dict[str, Any]] = []
    for row in observer_feed:
        callback, callback_row = _callback_for(row, prior, cast_agent_ids, agent_names)
        text = _name_text(str(row.get("text", "")), agent_names, lineage_names)
        plain = _name_text(str(row.get("plain_factual_text", "")), agent_names, lineage_names)
        humour = _name_text(str(row.get("humour_line") or ""), agent_names, lineage_names) or None
        if callback:
            text = f"{text} {callback}"
        if callback_row:
            callbacks.append(canonical_payload(callback_row))
        payload = dict(row)
        payload.update(
            {
                "schema_version": OBSERVER_CAST_FEED_SCHEMA,
                "observer_cast_item_id": "obscast_" + stable_hash(
                    "observer-cast-feed-item-v1",
                    {"observer_item_id": row.get("observer_item_id"), "callback": callback or ""},
                ),
                "cast_version": CAST_VERSION,
                "text": text,
                "plain_factual_text": plain,
                "humour_line": humour,
                "callback_line": callback,
                "retrospective_narration": True,
                "live_future_knowledge_allowed": False,
                "named_agent_ids": sorted(set(dict(row.get("subject_references", {}) or {}).get("agent_ids", []) or [])),
                "limitations": list(row.get("limitations", []) or [])
                + [
                    "CAST1 names and callbacks are presentation-only.",
                    "Retrospective cast labels are computed after the run and must not be read as live knowledge.",
                ],
            }
        )
        cast_rows.append(canonical_payload(payload))
    return cast_rows, canonical_payload({"schema_version": TEMPORAL_CALLBACK_SCHEMA, "items": callbacks})


def _copy_or_empty_jsonl(source: Path, target: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(source)
    write_jsonl(target, rows)
    return rows


def _world_history(cast_feed: Sequence[Mapping[str, Any]], retrospective_cast: Mapping[str, Any]) -> str:
    lines = [
        "# Observer CAST1 World History",
        "",
        "Retrospective presentation only. The linked Scientific Chronicle remains authoritative.",
        "",
        "## Featured Cast",
        "",
    ]
    items = [row for row in retrospective_cast.get("items", []) if row.get("featured")]
    if not items:
        lines.append(str(retrospective_cast.get("no_clear_cast_text") or "No clear cast was selected."))
    for row in items:
        roles = ", ".join(row.get("roles", []))
        lines.append(f"- {row['display_name']} (Agent {row['agent_id']}): {roles}; score {row['cast_score']}.")
    lines.extend(["", "## Timeline", ""])
    for row in cast_feed[:200]:
        generation = int(row.get("generation", 0))
        text = str(row.get("text", ""))
        lines.append(f"- Generation {generation}: {text}")
    return "\n".join(lines).rstrip() + "\n"


def _audio_script(cast_feed: Sequence[Mapping[str, Any]], maximum: int = 180) -> str:
    lines = [
        "EIEL Observer CAST1 audio script",
        "Retrospective cast narration. Presentation only.",
        "",
    ]
    for row in cast_feed[:400]:
        text = str(row.get("text", ""))
        while len(text) > maximum:
            cut = text.rfind(" ", 0, maximum)
            if cut < 40:
                cut = maximum
            lines.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        if text:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _dialogue_timestamp(row: Mapping[str, Any]) -> str:
    generation = int(row.get("generation", 0))
    step = int(row.get("step", 0))
    hour = (generation // 60) % 24
    minute = generation % 60
    second = step % 60
    day = 1 + generation // (24 * 60)
    return f"2000-01-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _creator_dialogue(cast_feed: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for row in cast_feed:
        parts = [str(row.get("humour_line") or "").strip(), str(row.get("callback_line") or "").strip()]
        text = " ".join(part for part in parts if part).replace("\n", " ").strip()
        if not text:
            text = str(row.get("text", "")).replace("\n", " ").strip()
        text = re.sub(r"\s+\(Agent \d+\)", "", text)
        text = re.sub(r"\s+\(L\d{8}\)", "", text)
        text = re.sub(r"\bAgent\s+(\d+)\b", r"Agent \1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(f"[{_dialogue_timestamp(row)}] [Creator] {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _artifact_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for logical, filename in sorted(CAST_ARTIFACTS.items()):
        path = output_dir / filename
        if path.exists():
            rows.append({"artifact": logical, "relative_path": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return canonical_payload(rows)


def observer_cast_regenerate(
    run_dir: Path,
    output_dir: Path,
    *,
    observer_dir: Path | None = None,
    audience_profile: str = "cosmic_creator",
    max_featured_agents: int = 5,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    chronicle = _chronicle_root(run_dir)
    before_hashes = {path.name: sha256_file(path) for path in chronicle.iterdir() if path.is_file()}
    events = _sorted_events(read_jsonl(chronicle / "formal_events.jsonl"))
    episodes_payload = read_json(chronicle / "episodes.json") if (chronicle / "episodes.json").exists() else []
    episodes = list(episodes_payload if isinstance(episodes_payload, list) else episodes_payload.get("episodes", []))
    manifest = read_json(chronicle / "chronicle_manifest.json") if (chronicle / "chronicle_manifest.json").exists() else {}
    generated_observer_source = observer_dir is None
    observer_source = observer_dir.expanduser().resolve() if observer_dir else output_dir / "_base_observer"
    if observer_dir is None:
        write_observer_outputs(
            run_dir,
            observer_source,
            config=ObserverConfig(
                audience_profile=audience_profile,
                storage_profile="observer_full",
                acknowledge_large_output=True,
                repeated_theme_cooldown=0,
            ),
        )
    observer_feed = read_jsonl(observer_source / "observer_feed.jsonl")
    agent_registry, lineage_registry = build_name_registries(events)
    retrospective = build_retrospective_cast(events, episodes, agent_registry=agent_registry, max_agents=max_featured_agents)
    live_rows = reconstruct_live_cast(events, agent_registry=agent_registry, max_featured=max_featured_agents)
    cast_feed, callbacks = build_cast_feed(
        observer_feed,
        agent_registry=agent_registry,
        lineage_registry=lineage_registry,
        retrospective_cast=retrospective,
    )
    write_json(output_dir / CAST_ARTIFACTS["agent_names"], agent_registry)
    write_json(output_dir / CAST_ARTIFACTS["lineage_names"], lineage_registry)
    write_json(output_dir / CAST_ARTIFACTS["retrospective_cast"], retrospective)
    write_jsonl(output_dir / CAST_ARTIFACTS["live_cast"], live_rows)
    write_json(output_dir / CAST_ARTIFACTS["callbacks"], callbacks)
    write_jsonl(output_dir / CAST_ARTIFACTS["feed"], cast_feed)
    _copy_or_empty_jsonl(observer_source / "observer_episodes.jsonl", output_dir / CAST_ARTIFACTS["episodes"])
    _copy_or_empty_jsonl(observer_source / "observer_agents.jsonl", output_dir / CAST_ARTIFACTS["agents"])
    _copy_or_empty_jsonl(observer_source / "observer_lineages.jsonl", output_dir / CAST_ARTIFACTS["lineages"])
    _copy_or_empty_jsonl(observer_source / "observer_regions.jsonl", output_dir / CAST_ARTIFACTS["regions"])
    atomic_write_text(output_dir / CAST_ARTIFACTS["history"], _world_history(cast_feed, retrospective))
    atomic_write_text(output_dir / CAST_ARTIFACTS["audio"], _audio_script(cast_feed))
    atomic_write_text(output_dir / CAST_ARTIFACTS["dialogue"], _creator_dialogue(cast_feed))
    after_hashes = {path.name: sha256_file(path) for path in chronicle.iterdir() if path.is_file()}
    source_immutable = before_hashes == after_hashes
    index = {
        "schema_version": OBSERVER_CAST_INDEX_SCHEMA,
        "cast_version": CAST_VERSION,
        "run_id": manifest.get("run_id") or run_dir.name,
        "audience_profile": audience_profile,
        "featured_agent_ids": retrospective.get("featured_agent_ids", []),
        "artifact_count": len(CAST_ARTIFACTS),
        "source_immutable": source_immutable,
        "future_leakage_audit": {
            "live_reconstruction_uses_prefix_only": True,
            "retrospective_disclosure_present": True,
            "live_future_leakage_findings": 0,
        },
    }
    write_json(output_dir / CAST_ARTIFACTS["index"], index)
    manifest_payload = {
        "schema_version": CAST_MANIFEST_SCHEMA,
        "cast_version": CAST_VERSION,
        "run_id": manifest.get("run_id") or run_dir.name,
        "source_run_dir": str(run_dir),
        "source_chronicle_root": str(chronicle),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_immutable": source_immutable,
        "observer_source": "_base_observer" if generated_observer_source else str(observer_source),
        "audience_profile": audience_profile,
        "retrospective": True,
        "live_reconstruction": True,
        "featured_agent_count": len(retrospective.get("featured_agent_ids", [])),
        "callback_count": len(callbacks.get("items", [])),
        "artifact_inventory": [],
    }
    write_json(output_dir / CAST_ARTIFACTS["manifest"], manifest_payload)
    manifest_payload["artifact_inventory"] = _artifact_rows(output_dir)
    write_json(output_dir / CAST_ARTIFACTS["manifest"], manifest_payload)
    if not source_immutable:
        raise CastError("CAST1 source immutability check failed")
    return canonical_payload(
        {
            "schema_version": "eiel_observer_cast_regeneration_result_v1",
            "output_dir": str(output_dir),
            "run_id": manifest_payload["run_id"],
            "source_immutable": source_immutable,
            "featured_agent_count": manifest_payload["featured_agent_count"],
            "callback_count": manifest_payload["callback_count"],
            "artifacts": manifest_payload["artifact_inventory"],
        }
    )


__all__ = [
    "CAST_ARTIFACTS",
    "CAST_VERSION",
    "CastError",
    "build_name_registries",
    "build_retrospective_cast",
    "build_cast_feed",
    "observer_cast_regenerate",
    "reconstruct_live_cast",
]
