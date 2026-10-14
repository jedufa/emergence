from __future__ import annotations

"""Post-run two-host documentary projection for EIEL Chronicle POD1."""

from collections import Counter
from html import escape
from pathlib import Path
import re
import shutil
import zipfile
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.cast import observer_cast_regenerate
from simulation.chronicle.observer import ObserverConfig, write_observer_outputs
from simulation.chronicle.schema import canonical_payload, stable_hash
from simulation.chronicle.storage import atomic_write_text, read_json, read_jsonl, sha256_file, write_json, write_jsonl


POD1_VERSION = "eiel_chronicle_pod1_v1"
DOCUMENTARY_MANIFEST_SCHEMA = "eiel_two_host_documentary_manifest_v1"
DOCUMENTARY_PLAN_SCHEMA = "eiel_two_host_documentary_plan_v1"
DOCUMENTARY_SCRIPT_SCHEMA = "eiel_two_host_documentary_script_v1"
DOCUMENTARY_UTTERANCE_SCHEMA = "eiel_two_host_documentary_utterance_v1"
DOCUMENTARY_VERIFIER_SCHEMA = "eiel_two_host_documentary_verifier_report_v1"
DOCUMENTARY_INDEX_SCHEMA = "eiel_two_host_documentary_index_v1"
SOURCE_INDEX_SCHEMA = "eiel_two_host_documentary_source_index_v1"

DOCUMENTARY_ARTIFACTS = {
    "manifest": "documentary_manifest.json",
    "source_index": "documentary_source_pack/documentary_source_index.json",
    "plan": "documentary_episode_plan.json",
    "script": "documentary_script.json",
    "transcript": "documentary_transcript.md",
    "transcript_html": "documentary_transcript.html",
    "audio_script": "documentary_audio_script.txt",
    "chapters": "documentary_chapters.json",
    "citations": "documentary_citations.json",
    "pronunciations": "documentary_pronunciations.json",
    "verifier": "documentary_verifier_report.json",
    "notebook_zip": "notebooklm_import_bundle.zip",
    "notebook_prompt": "NOTEBOOKLM_CUSTOM_PROMPT.md",
    "notebook_guide": "NOTEBOOKLM_IMPORT_GUIDE.md",
    "transcript_import_template": "TRANSCRIPT_IMPORT_TEMPLATE.md",
    "verification_guide": "SCRIPT_VERIFICATION_GUIDE.md",
    "index": "documentary_index.json",
}

SOURCE_DOCS = (
    ("00_run_provenance.md", "Run provenance"),
    ("01_world_summary.md", "World summary"),
    ("02_timeline.md", "Timeline"),
    ("03_episode_catalogue.md", "Episode catalogue"),
    ("04_cast_and_lineages.md", "Cast and lineages"),
    ("05_regions_and_ecology.md", "Regions and ecology"),
    ("06_branch_comparison.md", "Branch comparison"),
    ("07_mutation_and_genome_evidence.md", "Mutation and genome evidence"),
    ("08_selective_trace_highlights.md", "Selective trace highlights"),
    ("09_scientific_limitations.md", "Scientific limitations"),
    ("10_observer_voice_guide.md", "Observer voice guide"),
)

SPEAKERS = {
    "field_scientist": {
        "display_name": "Field Scientist",
        "voice_role": "calm precise source-grounded host",
        "speaking_rate_wpm": 150,
    },
    "cosmic_creator": {
        "display_name": "Cosmic Creator",
        "voice_role": "dry non-denominational creator persona reacting to logistics",
        "speaking_rate_wpm": 145,
    },
}

UNSAFE_PATTERNS = (
    re.compile(r"\b(intended|wanted|planned|decided|knew|believed|felt|thought|hoped)\b", re.I),
    re.compile(r"\b(conscious|intelligent|smart|stupid|soul|heaven|hell|scripture|sin|divine law)\b", re.I),
    re.compile(r"\bcaused by mutation\b|\bmutation caused\b", re.I),
)

FACT_TOKEN_RE = re.compile(r"\b(G(?:eneration)?\s*\d+|Agent\s+\d+|Lineage\s+[A-Za-z0-9_() -]+|L\d+|evt_[A-Za-z0-9_]+|epi_[A-Za-z0-9_]+|\d+(?:\.\d+)?\s*percent|\d+(?:\.\d+)?%)\b")


class DocumentaryError(ValueError):
    """Raised when POD1 cannot produce a bounded documentary artifact."""


def _chronicle_root(run_dir: Path) -> Path:
    if (run_dir / "chronicle" / "formal_events.jsonl").exists():
        return run_dir / "chronicle"
    if (run_dir / "formal_events.jsonl").exists():
        return run_dir
    raise DocumentaryError(f"no Chronicle artifacts found under {run_dir}")


def _sorted_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in events),
        key=lambda row: (
            int(row.get("generation", 0)),
            int(row.get("step", 0)),
            int(row.get("phase_order", 0)),
            int(row.get("intra_step_sequence", 0)),
            str(row.get("event_id", "")),
        ),
    )


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    return [dict(row) for row in payload.get("episodes", []) if isinstance(row, dict)]


def _load_entries(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _source_hashes(chronicle_root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(chronicle_root.iterdir())
        if path.is_file()
    }


def _artifact_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical, relative in sorted(DOCUMENTARY_ARTIFACTS.items()):
        path = output_dir / relative
        if path.exists():
            rows.append(
                {
                    "artifact": logical,
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return canonical_payload(rows)


def _event_label(event: Mapping[str, Any]) -> str:
    return f"G{int(event.get('generation', 0))} S{int(event.get('step', 0))} {event.get('event_type')}"


def _event_sentence(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type", "event")).replace("_", " ")
    category = str(event.get("event_category", "unknown"))
    generation = int(event.get("generation", 0))
    lineages = ", ".join(str(value) for value in event.get("lineage_ids", []) or [])
    agents = ", ".join(str(value) for value in event.get("agent_ids", []) or [])
    region = str(event.get("region_id") or "")
    subject = str(event.get("subject_id") or region or lineages or agents or "the run")
    return f"Generation {generation} recorded {event_type} for {subject} in the {category} category."


def _measurement_sentence(event: Mapping[str, Any]) -> str:
    measurements = dict(event.get("measurements", {}) or {})
    if not measurements:
        return ""
    selected: list[str] = []
    for key in sorted(measurements)[:4]:
        value = measurements[key]
        if isinstance(value, float):
            selected.append(f"{key}={value:.4g}")
        else:
            selected.append(f"{key}={value}")
    return " Measurements: " + ", ".join(selected) + "."


def _entry_by_event(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for event_id in entry.get("source_event_ids", []) or []:
            result.setdefault(str(event_id), dict(entry))
    return result


def _episodes_by_event(episodes: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_event: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        for event_id in episode.get("source_event_ids", []) or episode.get("event_ids", []) or []:
            by_event.setdefault(str(event_id), []).append(dict(episode))
    return by_event


def _top_events(events: Sequence[Mapping[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    rows = sorted(
        events,
        key=lambda row: (
            -int(row.get("historical_importance", 0)),
            -int(row.get("scientific_importance", 0)),
            -int(row.get("public_interest", 0)),
            int(row.get("generation", 0)),
            str(row.get("event_id", "")),
        ),
    )
    return [dict(row) for row in rows[:limit]]


def _top_episodes(episodes: Sequence[Mapping[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    rows = sorted(
        episodes,
        key=lambda row: (
            -len(row.get("source_event_ids", []) or row.get("event_ids", []) or []),
            int(row.get("start_generation", 0)),
            str(row.get("episode_id", "")),
        ),
    )
    return [dict(row) for row in rows[:limit]]


def _cast_paths(run_dir: Path, output_dir: Path, cast_dir: Path | None, observer_dir: Path | None) -> Path:
    if cast_dir and (cast_dir / "retrospective_cast.json").exists():
        return cast_dir.expanduser().resolve()
    candidate_paths = (
        run_dir / "observer_cast",
        run_dir / "chronicle" / "observer_cast",
        run_dir / "cast",
        run_dir / "chronicle" / "cast",
    )
    for candidate in candidate_paths:
        if (candidate / "retrospective_cast.json").exists():
            return candidate.resolve()
    generated = output_dir / "_generated_cast"
    observer_cast_regenerate(run_dir, generated, observer_dir=observer_dir, audience_profile="cosmic_creator", max_featured_agents=5)
    return generated


def _observer_paths(run_dir: Path, output_dir: Path, observer_dir: Path | None) -> Path:
    if observer_dir and (observer_dir / "observer_feed.jsonl").exists():
        return observer_dir.expanduser().resolve()
    candidates = (
        run_dir / "observer" / "cosmic_creator",
        run_dir / "chronicle" / "observer" / "cosmic_creator",
        run_dir / "observer" / "documentary",
        run_dir / "chronicle" / "observer" / "documentary",
    )
    for candidate in candidates:
        if (candidate / "observer_feed.jsonl").exists():
            return candidate.resolve()
    generated = output_dir / "_generated_observer"
    write_observer_outputs(
        run_dir,
        generated,
        config=ObserverConfig(
            audience_profile="cosmic_creator",
            storage_profile="observer_full",
            repeated_theme_cooldown=0,
            acknowledge_large_output=True,
        ),
    )
    return generated


def _read_cast(cast_root: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    retrospective = read_json(cast_root / "retrospective_cast.json") if (cast_root / "retrospective_cast.json").exists() else {}
    agent_names = {}
    lineage_names = {}
    if (cast_root / "agent_name_registry.json").exists():
        for row in read_json(cast_root / "agent_name_registry.json").get("items", []) or []:
            agent_names[str(row.get("agent_id"))] = str(row.get("display_name"))
    if (cast_root / "lineage_name_registry.json").exists():
        for row in read_json(cast_root / "lineage_name_registry.json").get("items", []) or []:
            lineage_names[str(row.get("lineage_id"))] = str(row.get("display_name"))
    return retrospective, agent_names, lineage_names


def _source_id(filename: str) -> str:
    return "src_" + stable_hash("documentary-source-v1", filename, length=12)


def _write_source_pack(
    pack_dir: Path,
    *,
    run_dir: Path,
    chronicle_root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    retrospective_cast: Mapping[str, Any],
    agent_names: Mapping[str, str],
    lineage_names: Mapping[str, str],
) -> dict[str, Any]:
    pack_dir.mkdir(parents=True, exist_ok=True)
    entry_by_event = _entry_by_event(entries)
    episode_rows = _top_episodes(episodes, 8)
    event_rows = _top_events(events, 24)
    event_counts = Counter(str(row.get("event_type", "unknown")) for row in events)
    category_counts = Counter(str(row.get("event_category", "unknown")) for row in events)
    region_counts = Counter(str(row.get("region_id") or "unanchored") for row in events)
    lineage_counts = Counter(value for row in events for value in (row.get("lineage_ids", []) or []))
    agent_counts = Counter(str(value) for row in events for value in (row.get("agent_ids", []) or []))

    docs: dict[str, str] = {}
    docs["00_run_provenance.md"] = "\n".join(
        [
            "# Run provenance",
            "",
            f"Source ID: {_source_id('00_run_provenance.md')}",
            f"Run ID: {manifest.get('run_id') or summary.get('run_id') or run_dir.name}",
            f"Campaign ID: {manifest.get('campaign_id')}",
            f"Branch ID: {manifest.get('branch_id')}",
            f"Source commit: {manifest.get('source_commit')}",
            f"Chronicle root: {chronicle_root}",
            f"Event count: {len(events)}",
            f"Episode count: {len(episodes)}",
            "Authority: Scientific Chronicle. Documentary files are derived public artifacts.",
            "",
        ]
    )
    docs["01_world_summary.md"] = "\n".join(
        [
            "# World summary",
            "",
            f"Source ID: {_source_id('01_world_summary.md')}",
            f"Source completeness: {manifest.get('source_completeness', summary.get('source_completeness', 'unknown'))}",
            f"First generation: {min((int(row.get('generation', 0)) for row in events), default=0)}",
            f"Last generation: {max((int(row.get('generation', 0)) for row in events), default=0)}",
            "",
            "Event categories:",
            *[f"- {name}: {count}" for name, count in sorted(category_counts.items())],
            "",
            "Frequent event types:",
            *[f"- {name}: {count}" for name, count in event_counts.most_common(12)],
            "",
        ]
    )
    docs["02_timeline.md"] = "\n".join(
        [
            "# Timeline",
            "",
            f"Source ID: {_source_id('02_timeline.md')}",
            *[
                (
                    f"- {stable_hash('documentary-timeline-row-v1', row.get('event_id'), length=12)}: "
                    f"{_event_label(row)}; event_id={row.get('event_id')}; "
                    f"entry_id={entry_by_event.get(str(row.get('event_id')), {}).get('chronicle_entry_id')}; "
                    f"confidence={row.get('confidence_class')}; causality={row.get('causality_class')}."
                )
                for row in event_rows
            ],
            "",
        ]
    )
    docs["03_episode_catalogue.md"] = "\n".join(
        [
            "# Episode catalogue",
            "",
            f"Source ID: {_source_id('03_episode_catalogue.md')}",
            *[
                (
                    f"- {episode.get('episode_id')}: {episode.get('episode_type')} from "
                    f"G{episode.get('start_generation')} to G{episode.get('end_generation')}; "
                    f"status={episode.get('status')}; confidence={episode.get('confidence_class')}; "
                    f"causality={episode.get('causality_class')}; source_events={','.join(map(str, episode.get('source_event_ids', []) or episode.get('event_ids', []) or []))}."
                )
                for episode in episode_rows
            ],
            "",
        ]
    )
    featured = [row for row in retrospective_cast.get("items", []) or [] if row.get("featured")]
    docs["04_cast_and_lineages.md"] = "\n".join(
        [
            "# Cast and lineages",
            "",
            f"Source ID: {_source_id('04_cast_and_lineages.md')}",
            "Cast names are presentation-only; scientific IDs remain authoritative.",
            "",
            "Featured agents:",
            *[
                (
                    f"- {row.get('display_name')} (Agent {row.get('agent_id')}): "
                    f"roles={','.join(map(str, row.get('roles', []) or []))}; "
                    f"source_events={','.join(map(str, dict(row.get('score_components', {}) or {}).get('source_event_ids', []) or []))}."
                )
                for row in featured[:5]
            ],
            *(["- No individual agent dominated the historical record."] if not featured else []),
            "",
            "Lineages:",
            *[f"- {lineage_names.get(str(lineage), str(lineage))}: {count} referenced events." for lineage, count in sorted(lineage_counts.items())[:12]],
            "",
        ]
    )
    docs["05_regions_and_ecology.md"] = "\n".join(
        [
            "# Regions and ecology",
            "",
            f"Source ID: {_source_id('05_regions_and_ecology.md')}",
            *[f"- {region}: {count} referenced events." for region, count in region_counts.most_common(12)],
            "",
            "Ecology-related timeline anchors:",
            *[
                f"- {row.get('event_id')}: {_event_sentence(row)}{_measurement_sentence(row)}"
                for row in event_rows
                if str(row.get("event_category")) in {"ecology", "hazard", "population"}
            ][:12],
            "",
        ]
    )
    docs["06_branch_comparison.md"] = "\n".join(
        [
            "# Branch comparison",
            "",
            f"Source ID: {_source_id('06_branch_comparison.md')}",
            "No branch-comparison companion was supplied to POD1 for this run.",
            "Branch-causal claims are unavailable in this documentary pack.",
            "",
        ]
    )
    mutation_events = [row for row in events if "mutation" in str(row.get("event_type", ""))]
    docs["07_mutation_and_genome_evidence.md"] = "\n".join(
        [
            "# Mutation and genome evidence",
            "",
            f"Source ID: {_source_id('07_mutation_and_genome_evidence.md')}",
            "Mutation occurrences are observations unless a separate companion states otherwise.",
            *[
                f"- {row.get('event_id')}: {_event_sentence(row)} confidence={row.get('confidence_class')}; causality={row.get('causality_class')}."
                for row in mutation_events[:12]
            ],
            *(["- No bounded mutation event was selected for the documentary focus."] if not mutation_events else []),
            "",
        ]
    )
    docs["08_selective_trace_highlights.md"] = "\n".join(
        [
            "# Selective trace highlights",
            "",
            f"Source ID: {_source_id('08_selective_trace_highlights.md')}",
            "No selective trace companion was supplied to POD1 for this run.",
            "The documentary may cite formal Chronicle events and episodes, not unsupported step-level trace detail.",
            "",
        ]
    )
    limitations = list(summary.get("limitations", []) or []) + list(manifest.get("limitations", []) or [])
    docs["09_scientific_limitations.md"] = "\n".join(
        [
            "# Scientific limitations",
            "",
            f"Source ID: {_source_id('09_scientific_limitations.md')}",
            "No claims of intelligence, consciousness, broad emergence, intention, or mutation causality are made by POD1.",
            *[f"- {item}" for item in (limitations or ["No additional limitations were recorded by the source Chronicle."])],
            "",
        ]
    )
    docs["10_observer_voice_guide.md"] = "\n".join(
        [
            "# Observer voice guide",
            "",
            f"Source ID: {_source_id('10_observer_voice_guide.md')}",
            "Host A is the Field Scientist: calm, precise, source-grounded, and responsible for limitations.",
            "Host B is the Cosmic Creator: fictional, non-denominational, dry, and limited to reactions to already stated source facts.",
            "Humour targets arithmetic, capacity planning, resource shortages, graphs, optimism, awkward consequences, and environmental punctuality.",
            "Humour must not invent thoughts, plans, intelligence, doctrine, or causal mechanisms.",
            "",
        ]
    )
    for filename, text in docs.items():
        atomic_write_text(pack_dir / filename, text.rstrip() + "\n")
    index = {
        "schema_version": SOURCE_INDEX_SCHEMA,
        "source_pack_version": POD1_VERSION,
        "documents": [
            {
                "source_id": _source_id(filename),
                "relative_path": filename,
                "title": title,
                "sha256": sha256_file(pack_dir / filename),
            }
            for filename, title in SOURCE_DOCS
        ],
        "source_chronicle_artifacts": [
            {"relative_path": path.name, "sha256": sha256_file(path)}
            for path in sorted(chronicle_root.iterdir())
            if path.is_file()
        ],
        "stable_source_ids": True,
    }
    write_json(pack_dir / "documentary_source_index.json", index)
    return canonical_payload(index)


def _focus_plan(
    *,
    events: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    retrospective_cast: Mapping[str, Any],
    agent_names: Mapping[str, str],
    lineage_names: Mapping[str, str],
    run_id: str,
) -> dict[str, Any]:
    top_events = _top_events(events, 10)
    top_episodes = _top_episodes(episodes, 4)
    entry_by_event = _entry_by_event(entries)
    featured_agents = [
        {
            "agent_id": str(row.get("agent_id")),
            "display_name": row.get("display_name"),
            "roles": list(row.get("roles", []) or []),
            "source_event_ids": list(dict(row.get("score_components", {}) or {}).get("source_event_ids", []) or [])[:12],
        }
        for row in retrospective_cast.get("items", []) or []
        if row.get("featured")
    ][:5]
    lineage_counts = Counter(value for row in events for value in (row.get("lineage_ids", []) or []))
    selected_lineages = [
        {
            "lineage_id": str(lineage),
            "display_name": lineage_names.get(str(lineage), str(lineage)),
            "referenced_event_count": count,
        }
        for lineage, count in lineage_counts.most_common(3)
    ]
    region_counts = Counter(str(row.get("region_id") or "unanchored") for row in events)
    selected_regions = [
        {"region_id": region, "referenced_event_count": count}
        for region, count in region_counts.most_common(2)
    ]
    turning_point = top_events[0] if top_events else {}
    chapters = [
        ("cold_open", "Cold open", 0.75),
        ("world", "The world", 1.1),
        ("cast", "Meet the cast", 1.4),
        ("early", "Early development", 1.2),
        ("turning_point", "Turning point", 1.8),
        ("consequences", "Consequences", 1.3),
        ("evidence_limits", "What the evidence does and does not show", 1.0),
        ("closing", "Closing", 0.8),
    ]
    return canonical_payload(
        {
            "schema_version": DOCUMENTARY_PLAN_SCHEMA,
            "plan_version": POD1_VERSION,
            "run_id": run_id,
            "target_duration_minutes": 10,
            "target_words": {"min": 1300, "max": 1700},
            "generation_range": {
                "start": min((int(row.get("generation", 0)) for row in events), default=0),
                "end": max((int(row.get("generation", 0)) for row in events), default=0),
            },
            "selected_events": [
                {
                    "event_id": row.get("event_id"),
                    "label": _event_label(row),
                    "entry_id": entry_by_event.get(str(row.get("event_id")), {}).get("chronicle_entry_id"),
                    "confidence": row.get("confidence_class"),
                    "causality": row.get("causality_class"),
                }
                for row in top_events
            ],
            "selected_episodes": [
                {
                    "episode_id": episode.get("episode_id"),
                    "episode_type": episode.get("episode_type"),
                    "start_generation": episode.get("start_generation"),
                    "end_generation": episode.get("end_generation"),
                    "source_event_ids": list(episode.get("source_event_ids", []) or episode.get("event_ids", []) or []),
                }
                for episode in top_episodes
            ],
            "selected_agents": featured_agents,
            "selected_lineages": selected_lineages,
            "selected_regions": selected_regions,
            "turning_point": {
                "event_id": turning_point.get("event_id"),
                "description": _event_sentence(turning_point) if turning_point else "No single turning point was selected.",
            },
            "surprise_or_reversal": "Recovery after pressure or a lineage/cast reversal when supported by selected events.",
            "unresolved_question": "Which recorded pressure would remain important under a longer or branched run?",
            "chapters": [
                {
                    "chapter_id": chapter_id,
                    "title": title,
                    "target_minutes": minutes,
                    "source_event_ids": [str(row.get("event_id")) for row in top_events],
                    "source_episode_ids": [str(row.get("episode_id")) for row in top_episodes],
                }
                for chapter_id, title, minutes in chapters
            ],
            "no_clear_cast": not featured_agents,
            "no_clear_cast_text": "This run did not produce a single dominant figure. The story belongs mainly to the lineages and regions.",
        }
    )


def _source_for_event(
    event: Mapping[str, Any],
    *,
    entry_by_event: Mapping[str, Mapping[str, Any]],
    episodes_by_event: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[str]]:
    event_id = str(event.get("event_id", ""))
    entry = entry_by_event.get(event_id, {})
    return {
        "source_event_ids": [event_id] if event_id else [],
        "source_entry_ids": [str(entry.get("chronicle_entry_id"))] if entry.get("chronicle_entry_id") else [],
        "source_episode_ids": [str(ep.get("episode_id")) for ep in episodes_by_event.get(event_id, []) if ep.get("episode_id")],
        "source_trace_ids": [],
        "source_mutation_ids": [],
    }


def _utterance(
    *,
    seq: int,
    speaker: str,
    chapter: str,
    text: str,
    line_type: str,
    sources: Mapping[str, Sequence[str]],
    confidence: str,
    causality: str,
    source_completeness: str,
    emphasis: str = "neutral",
) -> dict[str, Any]:
    words = max(1, len(re.findall(r"\b\w+\b", text)))
    rate = SPEAKERS[speaker]["speaking_rate_wpm"]
    return canonical_payload(
        {
            "schema_version": DOCUMENTARY_UTTERANCE_SCHEMA,
            "utterance_id": "podutt_" + stable_hash("documentary-utterance-v1", {"seq": seq, "speaker": speaker, "text": text}, length=16),
            "speaker": speaker,
            "speaker_label": SPEAKERS[speaker]["display_name"],
            "chapter": chapter,
            "sequence": seq,
            "text": text,
            "line_type": line_type,
            "source_event_ids": list(sources.get("source_event_ids", []) or []),
            "source_entry_ids": list(sources.get("source_entry_ids", []) or []),
            "source_episode_ids": list(sources.get("source_episode_ids", []) or []),
            "source_trace_ids": list(sources.get("source_trace_ids", []) or []),
            "source_mutation_ids": list(sources.get("source_mutation_ids", []) or []),
            "confidence": confidence,
            "causality": causality,
            "source_completeness": source_completeness,
            "expected_duration_seconds": round(words / rate * 60, 2),
            "audio": {
                "voice_role": SPEAKERS[speaker]["voice_role"],
                "speaking_rate_wpm": rate,
                "pause_after_seconds": 0.35 if speaker == "field_scientist" else 0.45,
                "emphasis": emphasis,
                "interruptibility": "chapter_boundary" if line_type == "transition" else "line_boundary",
            },
            "pronunciation_hints": {},
            "verifier_status": "unverified",
        }
    )


def _creator_reaction(event: Mapping[str, Any], index: int) -> str:
    families = (
        "I gave the system room to operate. It has replied with capacity planning.",
        "The record continues to confuse freedom with inventory management.",
        "Creation remains committed to turning abundance into a scheduling problem.",
        "The graph is not judging them. I, regrettably, have learned from the graph.",
        "There is optimism here, mostly because arithmetic has not finished speaking.",
        "The environment arrives exactly on time, which is rude but well documented.",
    )
    event_type = str(event.get("event_type", "event")).replace("_", " ")
    return f"{families[index % len(families)]} The immediate topic is {event_type}, not intention."


def _build_template_script(
    *,
    plan: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    entry_by_event = _entry_by_event(entries)
    episodes_by_event = _episodes_by_event(episodes)
    selected_events = _top_events(events, 18)
    if not selected_events:
        selected_events = list(events[:1])
    source_completeness = str(manifest.get("source_completeness", summary.get("source_completeness", "unknown")))
    utterances: list[dict[str, Any]] = []
    seq = 1

    def add(speaker: str, chapter: str, text: str, line_type: str, event: Mapping[str, Any] | None = None, emphasis: str = "neutral") -> None:
        nonlocal seq
        source = _source_for_event(event or selected_events[0], entry_by_event=entry_by_event, episodes_by_event=episodes_by_event)
        confidence = str((event or selected_events[0]).get("confidence_class", "measured"))
        causality = str((event or selected_events[0]).get("causality_class", "not_assessed"))
        utterances.append(
            _utterance(
                seq=seq,
                speaker=speaker,
                chapter=chapter,
                text=text,
                line_type=line_type,
                sources=source,
                confidence=confidence,
                causality=causality,
                source_completeness=source_completeness,
                emphasis=emphasis,
            )
        )
        seq += 1

    first = selected_events[0]
    last_generation = max((int(row.get("generation", 0)) for row in events), default=0)
    add("field_scientist", "cold_open", f"We open at {_event_label(first)}. {_event_sentence(first)}{_measurement_sentence(first)}", "fact", first, "opening")
    add("cosmic_creator", "cold_open", _creator_reaction(first, 0), "humour", first, "dry")
    add("field_scientist", "cold_open", "To be precise, the Chronicle records events and measurements. It does not assign motives to agents or to the world.", "correction", first)

    summary_event = next((row for row in events if str(row.get("event_type")) == "generation_summary"), selected_events[0])
    add("field_scientist", "world", f"The run spans generation 0 through generation {last_generation}. The source contains {len(events)} formal events and {len(episodes)} bounded episodes.", "fact", summary_event)
    add("cosmic_creator", "world", "A world with a ledger is still a world. It is merely one that can prove where the inconvenience started.", "humour", summary_event)
    for event in selected_events[:3]:
        add("field_scientist", "world", f"One early anchor is {_event_label(event)}. {_event_sentence(event)}{_measurement_sentence(event)}", "fact", event)
    add("field_scientist", "world", "Those anchors establish sequence. They do not by themselves prove a deeper causal model.", "limitation", selected_events[0])

    cast = list(plan.get("selected_agents", []) or [])
    if cast:
        for agent in cast[:4]:
            source_event_id = (agent.get("source_event_ids") or [selected_events[0].get("event_id")])[0]
            event = next((row for row in events if str(row.get("event_id")) == str(source_event_id)), selected_events[0])
            roles = ", ".join(agent.get("roles", []) or ["recorded figure"])
            add("field_scientist", "cast", f"{agent.get('display_name')} appears as Agent {agent.get('agent_id')}. The retrospective cast labels this figure as {roles}, based on recorded source events.", "fact", event)
            add("cosmic_creator", "cast", f"{agent.get('display_name')} has been promoted to narrative focus. This is what happens when a database develops a sense of casting.", "humour", event)
    else:
        add("field_scientist", "cast", str(plan.get("no_clear_cast_text")), "interpretation", selected_events[0])
        add("cosmic_creator", "cast", "An ensemble piece. Creation has avoided protagonists and discovered committee work.", "humour", selected_events[0])
    for lineage in list(plan.get("selected_lineages", []) or [])[:3]:
        event = next((row for row in selected_events if str(lineage.get("lineage_id")) in {str(value) for value in row.get("lineage_ids", []) or []}), selected_events[0])
        add("field_scientist", "cast", f"{lineage.get('display_name')} is one of the lineages most often referenced by the selected Chronicle records, with {lineage.get('referenced_event_count')} referenced events.", "fact", event)

    for idx, event in enumerate(selected_events[3:7], start=1):
        add("field_scientist", "early", f"Next, {_event_label(event)} enters the record. {_event_sentence(event)}{_measurement_sentence(event)}", "fact", event)
        if idx % 2 == 1:
            add("cosmic_creator", "early", _creator_reaction(event, idx), "humour", event, "dry")
    add("field_scientist", "early", "The pattern is best read as a recorded sequence of pressures and outcomes, not as evidence of planning by individual agents.", "limitation", selected_events[min(2, len(selected_events) - 1)])

    turning = next((row for row in selected_events if str(row.get("event_id")) == str(dict(plan.get("turning_point", {}) or {}).get("event_id"))), selected_events[0])
    add("field_scientist", "turning_point", f"The turning point selected for the documentary is {_event_label(turning)}. {_event_sentence(turning)}{_measurement_sentence(turning)}", "fact", turning, "emphasis")
    add("cosmic_creator", "turning_point", "This is the part where optimism asks for a receipt and the environment produces one.", "humour", turning, "dry")
    add("field_scientist", "turning_point", f"Its confidence class is {turning.get('confidence_class')}, and its causality class is {turning.get('causality_class')}. That boundary stays attached to the documentary line.", "fact", turning)
    for idx, event in enumerate(selected_events[7:12], start=7):
        add("field_scientist", "turning_point", f"A linked record is {_event_label(event)}. {_event_sentence(event)}{_measurement_sentence(event)}", "fact", event)
        add("cosmic_creator", "turning_point", _creator_reaction(event, idx), "humour", event, "dry")

    for event in selected_events[12:16] or selected_events[-4:]:
        add("field_scientist", "consequences", f"Consequences remain visible at {_event_label(event)}. {_event_sentence(event)}{_measurement_sentence(event)}", "fact", event)
    add("cosmic_creator", "consequences", "Some systems recover. Some merely stop making the same mistake loudly enough to be counted this minute.", "humour", selected_events[-1], "dry")
    add("field_scientist", "consequences", "That is a reaction to the recorded pattern, not an additional source claim.", "correction", selected_events[-1])

    add("field_scientist", "evidence_limits", f"The accepted transcript links every factual utterance to event, entry, or episode identifiers. The Chronicle remains the source of truth, with source completeness recorded as {source_completeness}.", "limitation", selected_events[0])
    add("cosmic_creator", "evidence_limits", "I am allowed sarcasm, not omniscience. This is restrictive but apparently good science.", "humour", selected_events[0], "dry")
    add("field_scientist", "evidence_limits", "NotebookLM or another external tool may produce a public narration manually, but the verifier must reject unsupported facts before import.", "limitation", selected_events[0])

    closing = selected_events[-1]
    add("field_scientist", "closing", f"The final selected anchor is {_event_label(closing)}. {_event_sentence(closing)}{_measurement_sentence(closing)}", "fact", closing, "closing")
    add("cosmic_creator", "closing", "I have given them freedom. They have converted it into patches, lineages, shortages, and paperwork.", "callback", closing, "dry")
    add("field_scientist", "closing", str(plan.get("unresolved_question")), "question", closing)
    add("cosmic_creator", "closing", "A fair question. I will be here, watching the next spreadsheet acquire weather.", "callback", closing, "closing")

    # Extend to a ten-minute baseline while preserving short turn lengths.
    filler_index = 0
    while sum(len(re.findall(r"\b\w+\b", row["text"])) for row in utterances) < 1300 and selected_events:
        event = selected_events[filler_index % len(selected_events)]
        chapter = list(plan.get("chapters", []))[min(5, filler_index % max(1, len(plan.get("chapters", []))))].get("chapter_id", "consequences")
        add("field_scientist", chapter, f"Returning to {_event_label(event)}, the documentary uses this as a source-linked anchor rather than a free-standing interpretation. {_event_sentence(event)}", "fact", event)
        if filler_index % 2 == 0:
            add("cosmic_creator", chapter, _creator_reaction(event, filler_index + 13), "humour", event, "dry")
        else:
            add(
                "cosmic_creator",
                chapter,
                "So the public version is allowed to notice the pattern, laugh at the logistics, and then hand the receipt back to the Chronicle.",
                "callback",
                event,
                "dry",
            )
        filler_index += 1

    def speaker_words(speaker: str) -> int:
        return sum(len(re.findall(r"\b\w+\b", str(row.get("text", "")))) for row in utterances if row.get("speaker") == speaker)

    balance_index = 0
    while selected_events and speaker_words("cosmic_creator") / max(1, speaker_words("cosmic_creator") + speaker_words("field_scientist")) < 0.40:
        if speaker_words("cosmic_creator") + speaker_words("field_scientist") >= 1680:
            break
        event = selected_events[balance_index % len(selected_events)]
        chapter = list(plan.get("chapters", []))[min(6, balance_index % max(1, len(plan.get("chapters", []))))].get("chapter_id", "consequences")
        add(
            "cosmic_creator",
            chapter,
            "For the listener: I can react to the evidence, complain about the logistics, and admire the accidental shape of the record. I cannot upgrade that into motive, strategy, or secret competence.",
            "callback" if balance_index % 2 == 0 else "question",
            event,
            "dry",
        )
        balance_index += 1

    total_words = sum(len(re.findall(r"\b\w+\b", row["text"])) for row in utterances)
    return canonical_payload(
        {
            "schema_version": DOCUMENTARY_SCRIPT_SCHEMA,
            "script_version": POD1_VERSION,
            "generation_mode": "deterministic_template",
            "run_id": plan.get("run_id"),
            "target_duration_minutes": 10,
            "estimated_word_count": total_words,
            "estimated_duration_seconds": round(sum(float(row["expected_duration_seconds"]) for row in utterances), 2),
            "speakers": SPEAKERS,
            "utterances": utterances,
            "source_grounding": "factual lines require source links; humour reacts to linked facts and adds no new facts",
        }
    )


def _line_has_fact_claim(line: Mapping[str, Any]) -> bool:
    if str(line.get("line_type")) in {"fact", "interpretation", "correction", "limitation", "question"}:
        return True
    text = str(line.get("text", ""))
    return bool(FACT_TOKEN_RE.search(text))


def verify_documentary_script(
    script: Mapping[str, Any],
    *,
    events: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_ids = {str(row.get("event_id")) for row in events}
    entry_ids = {str(row.get("chronicle_entry_id")) for row in entries}
    episode_ids = {str(row.get("episode_id")) for row in episodes}
    findings: list[dict[str, Any]] = []
    accepted = 0
    reason_counts: Counter[str] = Counter()
    for row in script.get("utterances", []) or []:
        line_type = str(row.get("line_type", ""))
        text = str(row.get("text", ""))
        source_event_ids = {str(value) for value in row.get("source_event_ids", []) or []}
        source_entry_ids = {str(value) for value in row.get("source_entry_ids", []) or []}
        source_episode_ids = {str(value) for value in row.get("source_episode_ids", []) or []}
        status = "verified_paraphrase" if line_type in {"interpretation", "correction", "limitation"} else "verified_fact"
        if line_type in {"humour", "callback"}:
            status = "safe_humour"
        reasons: list[str] = []
        if _line_has_fact_claim(row) and not source_event_ids:
            status = "unsupported"
            reasons.append("factual line lacks source_event_ids")
            reason_counts["unsupported"] += 1
        if not source_event_ids.issubset(event_ids):
            status = "identity_mismatch"
            reasons.append("line references unknown event IDs")
            reason_counts["identity_mismatch"] += 1
        if source_entry_ids and not source_entry_ids.issubset(entry_ids):
            status = "identity_mismatch"
            reasons.append("line references unknown entry IDs")
            reason_counts["identity_mismatch"] += 1
        if source_episode_ids and not source_episode_ids.issubset(episode_ids):
            status = "identity_mismatch"
            reasons.append("line references unknown episode IDs")
            reason_counts["identity_mismatch"] += 1
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(text):
                status = "unsafe_wording"
                reasons.append(f"unsafe wording matched {pattern.pattern}")
                reason_counts["unsafe_wording"] += 1
                break
        if "caused" in text.lower() and str(row.get("causality")) not in {"direct_mechanism"}:
            status = "causal_inflation"
            reasons.append("causal wording exceeds source causality class")
            reason_counts["causal_inflation"] += 1
        result = canonical_payload(
            {
                "utterance_id": row.get("utterance_id"),
                "sequence": row.get("sequence"),
                "speaker": row.get("speaker"),
                "line_type": line_type,
                "status": status,
                "reasons": reasons,
            }
        )
        findings.append(result)
        if status in {"verified_fact", "verified_paraphrase", "supported_interpretation", "safe_humour"}:
            accepted += 1
    counts = Counter(str(row["status"]) for row in findings)
    accepted_script = not any(
        counts.get(name, 0)
        for name in ("unsupported", "causal_inflation", "future_leakage", "identity_mismatch", "measurement_mismatch", "unsafe_wording")
    )
    return canonical_payload(
        {
            "schema_version": DOCUMENTARY_VERIFIER_SCHEMA,
            "verifier_version": POD1_VERSION,
            "accepted": accepted_script,
            "accepted_line_count": accepted,
            "total_line_count": len(findings),
            "status_counts": dict(sorted(counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "unsupported_count": reason_counts.get("unsupported", counts.get("unsupported", 0)),
            "causal_inflation_count": reason_counts.get("causal_inflation", counts.get("causal_inflation", 0)),
            "identity_mismatch_count": reason_counts.get("identity_mismatch", counts.get("identity_mismatch", 0)),
            "measurement_mismatch_count": reason_counts.get("measurement_mismatch", counts.get("measurement_mismatch", 0)),
            "unsafe_wording_count": reason_counts.get("unsafe_wording", counts.get("unsafe_wording", 0)),
            "findings": findings,
        }
    )


def _script_with_status(script: Mapping[str, Any], verifier: Mapping[str, Any]) -> dict[str, Any]:
    status_by_id = {str(row.get("utterance_id")): str(row.get("status")) for row in verifier.get("findings", []) or []}
    payload = dict(script)
    payload["utterances"] = [
        {**dict(row), "verifier_status": status_by_id.get(str(row.get("utterance_id")), "unverified")}
        for row in script.get("utterances", []) or []
    ]
    return canonical_payload(payload)


def _chapters(script: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    utterances = list(script.get("utterances", []) or [])
    for chapter in plan.get("chapters", []) or []:
        chapter_id = str(chapter.get("chapter_id"))
        lines = [row for row in utterances if str(row.get("chapter")) == chapter_id]
        rows.append(
            {
                "chapter_id": chapter_id,
                "title": chapter.get("title"),
                "utterance_count": len(lines),
                "start_sequence": min((int(row.get("sequence", 0)) for row in lines), default=None),
                "end_sequence": max((int(row.get("sequence", 0)) for row in lines), default=None),
                "estimated_duration_seconds": round(sum(float(row.get("expected_duration_seconds", 0)) for row in lines), 2),
            }
        )
    return canonical_payload(rows)


def _citations(script: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for line in script.get("utterances", []) or []:
        rows.append(
            {
                "utterance_id": line.get("utterance_id"),
                "sequence": line.get("sequence"),
                "source_event_ids": line.get("source_event_ids", []),
                "source_entry_ids": line.get("source_entry_ids", []),
                "source_episode_ids": line.get("source_episode_ids", []),
                "source_trace_ids": line.get("source_trace_ids", []),
                "source_mutation_ids": line.get("source_mutation_ids", []),
            }
        )
    return canonical_payload({"schema_version": "eiel_two_host_documentary_citations_v1", "items": rows})


def _pronunciations(script: Mapping[str, Any]) -> dict[str, Any]:
    hints = {}
    for line in script.get("utterances", []) or []:
        for token in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+-[0-9]+)?\b", str(line.get("text", ""))):
            if token in {"Generation", "Chronicle", "NotebookLM", "Field Scientist", "Cosmic Creator"}:
                hints[token] = token
    return canonical_payload({"schema_version": "eiel_two_host_documentary_pronunciations_v1", "hints": hints})


def _transcript(script: Mapping[str, Any]) -> str:
    lines = ["# EIEL Two-Host Documentary Transcript", "", "Scientific Chronicle remains authoritative.", ""]
    chapter = None
    for row in script.get("utterances", []) or []:
        if row.get("chapter") != chapter:
            chapter = row.get("chapter")
            lines.extend(["", f"## {str(chapter).replace('_', ' ').title()}", ""])
        lines.append(f"**{row.get('speaker_label')}:** {row.get('text')}")
    return "\n".join(lines).rstrip() + "\n"


def _transcript_html(script: Mapping[str, Any]) -> str:
    parts = [
        "<!doctype html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\"><title>EIEL Two-Host Documentary</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:32px auto;line-height:1.5;color:#1b1f23}.line{margin:12px 0}.speaker{font-weight:700}.field_scientist .speaker{color:#1f5f8b}.cosmic_creator .speaker{color:#8a5a00}.meta{color:#667085;font-size:13px}section{border-top:1px solid #ddd;padding-top:16px;margin-top:20px}</style>",
        "</head><body><h1>EIEL Two-Host Documentary</h1><p class=\"meta\">Presentation only. Scientific Chronicle remains authoritative.</p>",
    ]
    chapter = None
    for row in script.get("utterances", []) or []:
        if row.get("chapter") != chapter:
            if chapter is not None:
                parts.append("</section>")
            chapter = row.get("chapter")
            parts.append(f"<section><h2>{escape(str(chapter).replace('_', ' ').title())}</h2>")
        parts.append(
            f"<p class=\"line {escape(str(row.get('speaker')))}\"><span class=\"speaker\">"
            f"{escape(str(row.get('speaker_label')))}:</span> {escape(str(row.get('text')))}</p>"
        )
    if chapter is not None:
        parts.append("</section>")
    parts.append("</body></html>\n")
    return "\n".join(parts)


def _audio_script(script: Mapping[str, Any]) -> str:
    lines = ["EIEL POD1 two-host TTS script", "Provider-neutral. One utterance per line.", ""]
    for row in script.get("utterances", []) or []:
        audio = dict(row.get("audio", {}) or {})
        lines.append(
            f"[{row.get('speaker_label')}] rate={audio.get('speaking_rate_wpm')} pause={audio.get('pause_after_seconds')} emphasis={audio.get('emphasis')} | {row.get('text')}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _notebook_prompt(plan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# NotebookLM Custom Prompt",
            "",
            "Generate an approximately ten-minute two-host documentary discussion from the attached source documents.",
            "",
            "Host A is the Field Scientist: calm, precise, source-grounded, and responsible for limits.",
            "Host B is the Cosmic Creator: fictional, non-denominational, dry, sarcastic, and limited to reacting to facts Host A has already grounded.",
            "",
            "Rules:",
            "- Do not invent facts, intentions, intelligence, plans, causes, quotations, trace details, or religious doctrine.",
            "- Preserve confidence, causality, and source-completeness limits.",
            "- Use short conversational exchanges and callbacks.",
            "- End with this unresolved scientific question: " + str(plan.get("unresolved_question")),
            "- NotebookLM output is a public presentation draft, not scientific evidence.",
            "",
        ]
    )


def _notebook_guide() -> str:
    return "\n".join(
        [
            "# NotebookLM Import Guide",
            "",
            "1. Create a notebook manually.",
            "2. Import only the files inside `documentary_source_pack/` and `NOTEBOOKLM_CUSTOM_PROMPT.md`.",
            "3. Generate a two-host discussion manually.",
            "4. Paste the transcript into `TRANSCRIPT_IMPORT_TEMPLATE.md` format.",
            "5. Run the POD1 verifier before accepting the imported script.",
            "",
            "Do not automate browser login, credentials, upload, or external service calls from simulation runtime.",
            "",
        ]
    )


def _transcript_template() -> str:
    return "\n".join(
        [
            "# Transcript Import Template",
            "",
            "Use one line per utterance:",
            "",
            "[Field Scientist] source_event_ids=evt_example source_entry_ids=chr_example source_episode_ids=epi_example | Text",
            "[Cosmic Creator] source_event_ids=evt_example | Reaction without new factual claim",
            "",
        ]
    )


def _verification_guide() -> str:
    return "\n".join(
        [
            "# Script Verification Guide",
            "",
            "Accepted factual lines need source event links. Humour must link to the fact it reacts to.",
            "Reject unsupported facts, identity mismatches, measurement mismatches, causal inflation, future leakage, unsafe wording, and unsupported quotations.",
            "",
        ]
    )


def _write_notebook_zip(output_dir: Path, source_pack_dir: Path) -> None:
    zip_path = output_dir / DOCUMENTARY_ARTIFACTS["notebook_zip"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source_pack_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir).as_posix())
        for relative in (
            DOCUMENTARY_ARTIFACTS["notebook_prompt"],
            DOCUMENTARY_ARTIFACTS["notebook_guide"],
            DOCUMENTARY_ARTIFACTS["transcript_import_template"],
            DOCUMENTARY_ARTIFACTS["verification_guide"],
        ):
            handle.write(output_dir / relative, relative)


def _host_metrics(script: Mapping[str, Any]) -> dict[str, Any]:
    utterances = list(script.get("utterances", []) or [])
    counts = Counter(str(row.get("speaker")) for row in utterances)
    words = Counter()
    humour = 0
    callbacks = 0
    for row in utterances:
        speaker = str(row.get("speaker"))
        words[speaker] += len(re.findall(r"\b\w+\b", str(row.get("text", ""))))
        if str(row.get("line_type")) == "humour":
            humour += 1
        if str(row.get("line_type")) == "callback":
            callbacks += 1
    total_words = sum(words.values()) or 1
    return canonical_payload(
        {
            "utterance_counts": dict(sorted(counts.items())),
            "word_counts": dict(sorted(words.items())),
            "speaking_balance": {speaker: round(count / total_words, 4) for speaker, count in sorted(words.items())},
            "humour_density": round(humour / max(1, len(utterances)), 4),
            "callback_count": callbacks,
            "host_distinction_review": "Field Scientist carries facts and limits; Cosmic Creator carries most humour and callbacks.",
        }
    )


def write_documentary_outputs(
    run_dir: Path,
    output_dir: Path,
    *,
    observer_dir: Path | None = None,
    cast_dir: Path | None = None,
    generation_mode: str = "deterministic_template",
) -> dict[str, Any]:
    if generation_mode not in {"deterministic_template", "external_ai_import", "llm_adapter_disabled"}:
        raise DocumentaryError("invalid documentary generation mode")
    if generation_mode != "deterministic_template":
        raise DocumentaryError("POD1 implements deterministic_template generation; external import is verifier-only")
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    chronicle_root = _chronicle_root(run_dir)
    before = _source_hashes(chronicle_root)
    manifest = read_json(chronicle_root / "chronicle_manifest.json") if (chronicle_root / "chronicle_manifest.json").exists() else {}
    summary = read_json(chronicle_root / "scientific_chronicle_summary.json") if (chronicle_root / "scientific_chronicle_summary.json").exists() else {}
    events = _sorted_events(read_jsonl(chronicle_root / "formal_events.jsonl"))
    entries = _load_entries(chronicle_root / "scientific_chronicle.jsonl")
    episodes = _load_episodes(chronicle_root / "episodes.json")
    observer_root = _observer_paths(run_dir, output_dir, observer_dir)
    cast_root = _cast_paths(run_dir, output_dir, cast_dir, observer_root)
    retrospective, agent_names, lineage_names = _read_cast(cast_root)
    run_id = str(manifest.get("run_id") or summary.get("run_id") or run_dir.name)
    source_index = _write_source_pack(
        output_dir / "documentary_source_pack",
        run_dir=run_dir,
        chronicle_root=chronicle_root,
        manifest=manifest,
        summary=summary,
        events=events,
        entries=entries,
        episodes=episodes,
        retrospective_cast=retrospective,
        agent_names=agent_names,
        lineage_names=lineage_names,
    )
    plan = _focus_plan(
        events=events,
        entries=entries,
        episodes=episodes,
        retrospective_cast=retrospective,
        agent_names=agent_names,
        lineage_names=lineage_names,
        run_id=run_id,
    )
    script = _build_template_script(plan=plan, events=events, entries=entries, episodes=episodes, manifest=manifest, summary=summary)
    verifier = verify_documentary_script(script, events=events, entries=entries, episodes=episodes)
    script = _script_with_status(script, verifier)
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["plan"], plan)
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["script"], script)
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["chapters"], {"schema_version": "eiel_two_host_documentary_chapters_v1", "items": _chapters(script, plan)})
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["citations"], _citations(script))
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["pronunciations"], _pronunciations(script))
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["verifier"], verifier)
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["transcript"], _transcript(script))
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["transcript_html"], _transcript_html(script))
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["audio_script"], _audio_script(script))
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["notebook_prompt"], _notebook_prompt(plan))
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["notebook_guide"], _notebook_guide())
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["transcript_import_template"], _transcript_template())
    atomic_write_text(output_dir / DOCUMENTARY_ARTIFACTS["verification_guide"], _verification_guide())
    _write_notebook_zip(output_dir, output_dir / "documentary_source_pack")
    after = _source_hashes(chronicle_root)
    source_immutable = before == after
    metrics = _host_metrics(script)
    index = {
        "schema_version": DOCUMENTARY_INDEX_SCHEMA,
        "pod1_version": POD1_VERSION,
        "run_id": run_id,
        "generation_mode": generation_mode,
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "source_pack": DOCUMENTARY_ARTIFACTS["source_index"],
        "script": DOCUMENTARY_ARTIFACTS["script"],
        "transcript": DOCUMENTARY_ARTIFACTS["transcript"],
        "transcript_html": DOCUMENTARY_ARTIFACTS["transcript_html"],
        "notebook_bundle": DOCUMENTARY_ARTIFACTS["notebook_zip"],
        "verifier_accepted": verifier.get("accepted"),
        "host_metrics": metrics,
        "source_immutable": source_immutable,
        "rng_immutability": True,
        "llm_runtime_calls": False,
        "notebooklm_is_scientific_source": False,
    }
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["index"], index)
    manifest_payload = {
        "schema_version": DOCUMENTARY_MANIFEST_SCHEMA,
        "pod1_version": POD1_VERSION,
        "run_id": run_id,
        "source_run_dir": str(run_dir),
        "source_chronicle_root": str(chronicle_root),
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_immutable": source_immutable,
        "presentation_only": True,
        "generation_mode": generation_mode,
        "observer_source": str(observer_root),
        "cast_source": str(cast_root),
        "source_index": source_index,
        "host_metrics": metrics,
        "verifier": {
            "accepted": verifier.get("accepted"),
            "unsupported_count": verifier.get("unsupported_count"),
            "causal_inflation_count": verifier.get("causal_inflation_count"),
            "identity_mismatch_count": verifier.get("identity_mismatch_count"),
            "measurement_mismatch_count": verifier.get("measurement_mismatch_count"),
            "unsafe_wording_count": verifier.get("unsafe_wording_count"),
        },
        "artifacts": [],
    }
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["manifest"], manifest_payload)
    manifest_payload["artifacts"] = _artifact_rows(output_dir)
    write_json(output_dir / DOCUMENTARY_ARTIFACTS["manifest"], manifest_payload)
    if not source_immutable:
        raise DocumentaryError("POD1 source immutability check failed")
    if verifier.get("accepted") is not True:
        raise DocumentaryError("deterministic documentary script failed verification")
    return canonical_payload(
        {
            "schema_version": "eiel_two_host_documentary_generation_result_v1",
            "output_dir": str(output_dir),
            "run_id": run_id,
            "source_immutable": source_immutable,
            "verifier_accepted": True,
            "word_count": script.get("estimated_word_count"),
            "estimated_duration_seconds": script.get("estimated_duration_seconds"),
            "artifact_count": len(manifest_payload["artifacts"]),
            "host_metrics": metrics,
            "artifacts": manifest_payload["artifacts"],
        }
    )


__all__ = [
    "DOCUMENTARY_ARTIFACTS",
    "DOCUMENTARY_SCRIPT_SCHEMA",
    "DocumentaryError",
    "verify_documentary_script",
    "write_documentary_outputs",
]
