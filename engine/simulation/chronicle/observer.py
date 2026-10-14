from __future__ import annotations

"""Deterministic read-only public Observer companion for Scientific Chronicle."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.observer_templates import (
    AUDIENCE_PROFILES,
    CARD_HUMOUR_LINES,
    COSMIC_CARD_HUMOUR_LINES,
    COSMIC_HUMOUR_LINES,
    DRY_HUMOUR_LINES,
    EPISODE_HEADLINES,
    EVENT_HEADLINES,
    FORBIDDEN_CLAIM_PATTERNS,
    HUMOUR_CONTRACT,
    PROFILE_FRAMES,
    RELIGIOUS_SPECIFICITY_PATTERNS,
    STORAGE_PROFILES,
    template_inventory,
)
from simulation.chronicle.schema import (
    CHRONICLE_ENTRY_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    assert_unique_events,
    canonical_json,
    canonical_payload,
    event_order_key,
    stable_hash,
)
from simulation.chronicle.storage import atomic_write_text, sha256_file, write_json, write_jsonl


OBSERVER_VERSION = "eiel_chronicle_obs1_voice_v3"
OBSERVER_MANIFEST_SCHEMA = "eiel_observer_manifest_v1"
OBSERVER_FEED_SCHEMA = "eiel_observer_feed_item_v1"
OBSERVER_EPISODE_SCHEMA = "eiel_observer_episode_story_v1"
OBSERVER_AGENT_SCHEMA = "eiel_observer_agent_card_v1"
OBSERVER_LINEAGE_SCHEMA = "eiel_observer_lineage_card_v1"
OBSERVER_REGION_SCHEMA = "eiel_observer_region_card_v1"
OBSERVER_WORLD_SCHEMA = "eiel_observer_world_summary_v1"
OBSERVER_INDEX_SCHEMA = "eiel_observer_index_v1"
OBSERVER_PREFLIGHT_SCHEMA = "eiel_observer_storage_preflight_v1"

OBSERVER_ARTIFACTS = {
    "feed": "observer_feed.jsonl",
    "episodes": "observer_episodes.jsonl",
    "agents": "observer_agents.jsonl",
    "lineages": "observer_lineages.jsonl",
    "regions": "observer_regions.jsonl",
    "world_summary": "observer_world_summary.json",
    "world_history": "observer_world_history.md",
    "audio_script": "observer_audio_script.txt",
    "index": "observer_index.json",
    "templates": "observer_template_inventory.json",
    "preflight": "observer_storage_preflight.json",
    "manifest": "observer_manifest.json",
}

SCIENTIFIC_ARTIFACTS = {
    "events": "formal_events.jsonl",
    "entries": "scientific_chronicle.jsonl",
    "episodes": "episodes.json",
    "contexts": "context_windows.jsonl",
    "summary": "scientific_chronicle_summary.json",
    "manifest": "chronicle_manifest.json",
    "storage": "chronicle_storage_summary.json",
    "index": "index.json",
}

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "background": 4}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 512 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
_MAX_RECORDS = 1_000_000
_MAX_AUDIO_SENTENCE = 240
_TECHNICAL_CATEGORIES = frozenset({"integrity"})
_TECHNICAL_EVENT_TYPES = frozenset(
    {
        "chronicle_integrity",
        "chronicle_events_suppressed",
        "chronicle_storage_budget",
        "checkpoint_chronicle_metadata_mismatch",
        "retrospective_source_incomplete",
    }
)


class ObserverError(ValueError):
    """Raised when OBS1 cannot produce a safe, source-linked presentation."""


@dataclass(frozen=True)
class ObserverConfig:
    audience_profile: str = "public_plain"
    storage_profile: str = "observer_notable"
    minimum_public_interest: int = 0
    minimum_visual_interest: int = 0
    minimum_live_priority: str = "background"
    categories: tuple[str, ...] = field(default_factory=tuple)
    event_types: tuple[str, ...] = field(default_factory=tuple)
    episode_families: tuple[str, ...] = field(default_factory=tuple)
    generation_start: int | None = None
    generation_end: int | None = None
    agent_ids: tuple[int, ...] = field(default_factory=tuple)
    lineage_ids: tuple[str, ...] = field(default_factory=tuple)
    region_ids: tuple[str, ...] = field(default_factory=tuple)
    max_messages_per_generation: int = 8
    repeated_theme_cooldown: int = 1
    feed_budget: int = 10_000
    maximum_audio_sentence_length: int = 180
    acknowledge_large_output: bool = False

    def validate(self) -> None:
        if self.audience_profile not in AUDIENCE_PROFILES:
            raise ObserverError(f"unsupported audience profile: {self.audience_profile}")
        if self.storage_profile not in STORAGE_PROFILES:
            raise ObserverError(f"unsupported Observer storage profile: {self.storage_profile}")
        if self.minimum_live_priority not in _PRIORITY_RANK:
            raise ObserverError(f"unsupported minimum live priority: {self.minimum_live_priority}")
        for name in ("minimum_public_interest", "minimum_visual_interest"):
            value = int(getattr(self, name))
            if value < 0 or value > 5:
                raise ObserverError(f"{name} must be between 0 and 5")
        if self.generation_start is not None and self.generation_end is not None:
            if int(self.generation_start) > int(self.generation_end):
                raise ObserverError("generation_start must not exceed generation_end")
        if self.max_messages_per_generation < 1 or self.max_messages_per_generation > 10_000:
            raise ObserverError("max_messages_per_generation must be between 1 and 10000")
        if self.repeated_theme_cooldown < 0:
            raise ObserverError("repeated_theme_cooldown must be non-negative")
        if self.feed_budget < 1 or self.feed_budget > _MAX_RECORDS:
            raise ObserverError(f"feed_budget must be between 1 and {_MAX_RECORDS}")
        if self.maximum_audio_sentence_length < 60 or self.maximum_audio_sentence_length > _MAX_AUDIO_SENTENCE:
            raise ObserverError(
                f"maximum_audio_sentence_length must be between 60 and {_MAX_AUDIO_SENTENCE}"
            )
        if self.storage_profile == "observer_full" and not self.acknowledge_large_output:
            raise ObserverError("observer_full requires acknowledge_large_output=true")

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))


@dataclass(frozen=True)
class _ScientificSource:
    run_root: Path
    chronicle_root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    entries: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    hashes: list[dict[str, Any]]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _safe_input_root(value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ObserverError(f"source root must not be a symlink: {value}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ObserverError(f"source root is not a directory: {value}")
    return resolved


def _safe_output_root(value: Path, source: _ScientificSource) -> Path:
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ObserverError(f"output root must not be a symlink: {value}")
    resolved = expanded.resolve()
    if resolved == source.chronicle_root or _inside(resolved, source.chronicle_root):
        raise ObserverError("Observer output must be outside the scientific Chronicle directory")
    if resolved.exists() and not resolved.is_dir():
        raise ObserverError("Observer output path exists and is not a directory")
    return resolved


def _load_json(path: Path, expected: type | tuple[type, ...]) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ObserverError(f"missing or unsafe scientific artifact: {path.name}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ObserverError(f"scientific artifact exceeds bounded reader: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"invalid scientific artifact {path.name}: {exc}") from exc
    if not isinstance(value, expected):
        raise ObserverError(f"unexpected shape in scientific artifact: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ObserverError(f"missing or unsafe scientific artifact: {path.name}")
    if path.stat().st_size > _MAX_JSONL_BYTES:
        raise ObserverError(f"scientific feed exceeds bounded reader: {path.name}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if len(raw.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
                    raise ObserverError(f"scientific feed line exceeds bound: {path.name}:{line_number}")
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ObserverError(f"expected object in {path.name}:{line_number}")
                rows.append(value)
                if len(rows) > _MAX_RECORDS:
                    raise ObserverError(f"scientific feed exceeds {_MAX_RECORDS} records: {path.name}")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"invalid scientific feed {path.name}: {exc}") from exc
    return rows


def _scientific_hashes(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical_name, filename in sorted(SCIENTIFIC_ARTIFACTS.items()):
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ObserverError(f"missing or unsafe scientific artifact: {filename}")
        rows.append(
            {
                "artifact": logical_name,
                "relative_path": filename,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return canonical_payload(rows)


def load_scientific_source(source_root: Path) -> _ScientificSource:
    selected = _safe_input_root(source_root)
    nested = selected / "chronicle"
    chronicle_root = nested if (nested / SCIENTIFIC_ARTIFACTS["manifest"]).is_file() else selected
    if chronicle_root.is_symlink() or not _inside(chronicle_root, selected):
        raise ObserverError("scientific Chronicle path is unsafe")
    hashes = _scientific_hashes(chronicle_root)
    manifest = _load_json(chronicle_root / SCIENTIFIC_ARTIFACTS["manifest"], dict)
    summary = _load_json(chronicle_root / SCIENTIFIC_ARTIFACTS["summary"], dict)
    events = _load_jsonl(chronicle_root / SCIENTIFIC_ARTIFACTS["events"])
    entries = _load_jsonl(chronicle_root / SCIENTIFIC_ARTIFACTS["entries"])
    episodes = _load_json(chronicle_root / SCIENTIFIC_ARTIFACTS["episodes"], list)
    if str(manifest.get("schema_version")) != MANIFEST_SCHEMA_VERSION:
        raise ObserverError("unsupported scientific Chronicle manifest schema")
    if str(summary.get("schema_version")) != "eiel_scientific_chronicle_summary_v1":
        raise ObserverError("unsupported scientific Chronicle summary schema")
    if manifest.get("finalized") is not True:
        raise ObserverError("retrospective Observer generation requires a finalized Scientific Chronicle")
    assert_unique_events(events)
    if len(entries) != len(events):
        raise ObserverError("scientific event and entry feeds have different lengths")
    event_ids = {str(event["event_id"]) for event in events}
    episode_ids: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, dict) or str(episode.get("schema_version")) != EPISODE_SCHEMA_VERSION:
            raise ObserverError("unsupported scientific episode schema")
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id.startswith("epi_") or episode_id in episode_ids:
            raise ObserverError("scientific episode identity is missing or duplicated")
        episode_ids.add(episode_id)
        source_ids = {str(value) for value in episode.get("source_event_ids", []) or []}
        if not source_ids or not source_ids.issubset(event_ids):
            raise ObserverError("scientific episode references missing source events")
    run_id = str(manifest.get("run_id") or "")
    run_sequence: list[str] = []
    branch_sequence: list[str] = []
    entry_ids: set[str] = set()
    for event, entry in zip(events, entries):
        if str(entry.get("schema_version")) != CHRONICLE_ENTRY_SCHEMA_VERSION:
            raise ObserverError("unsupported scientific Chronicle entry schema")
        entry_id = str(entry.get("chronicle_entry_id") or "")
        if not entry_id.startswith("chr_") or entry_id in entry_ids:
            raise ObserverError("scientific Chronicle entry identity is missing or duplicated")
        entry_ids.add(entry_id)
        source_ids = {str(value) for value in entry.get("source_event_ids", []) or []}
        if str(event["event_id"]) not in source_ids or not source_ids.issubset(event_ids):
            raise ObserverError("scientific Chronicle entry is not traceable to its paired event")
        if entry.get("episode_id") and str(entry["episode_id"]) not in episode_ids:
            raise ObserverError("scientific Chronicle entry references a missing episode")
        event_run_id = str(event.get("run_id") or "")
        if event_run_id != str(entry.get("run_id") or ""):
            raise ObserverError("paired scientific event and entry run identities differ")
        event_branch_id = str(event.get("branch_id") or "")
        if not run_sequence or run_sequence[-1] != event_run_id:
            run_sequence.append(event_run_id)
        if not branch_sequence or branch_sequence[-1] != event_branch_id:
            branch_sequence.append(event_branch_id)
        manifest_campaign = str(manifest.get("campaign_id") or "")
        if manifest_campaign and str(event.get("campaign_id") or "") != manifest_campaign:
            raise ObserverError("scientific Chronicle contains a foreign campaign identity")
    if not run_sequence or run_sequence[-1] != run_id:
        raise ObserverError("scientific Chronicle does not end in the manifest run identity")
    if len(run_sequence) > 1:
        if not manifest.get("parent_checkpoint_id"):
            raise ObserverError("scientific Chronicle mixes run identities without checkpoint ancestry")
        if len(run_sequence) > 2 or len(set(run_sequence)) != len(run_sequence):
            raise ObserverError("scientific Chronicle has an ambiguous checkpoint run-identity sequence")
    manifest_branch = str(manifest.get("branch_id") or "")
    if manifest_branch and branch_sequence and branch_sequence[-1] != manifest_branch:
        raise ObserverError("scientific Chronicle does not end in the manifest branch identity")
    if len(branch_sequence) > 2 or len(set(branch_sequence)) != len(branch_sequence):
        raise ObserverError("scientific Chronicle has an ambiguous branch-identity sequence")
    if int(manifest.get("event_count", len(events))) != len(events):
        raise ObserverError("scientific manifest event count does not match the feed")
    if int(manifest.get("episode_count", len(episodes))) != len(episodes):
        raise ObserverError("scientific manifest episode count does not match the episode artifact")
    if summary.get("run_id") is not None and str(summary.get("run_id")) != run_id:
        raise ObserverError("scientific summary and manifest run identities differ")
    summary_counts = {
        "event_count": len(events),
        "chronicle_entry_count": len(entries),
        "episode_count": len(episodes),
    }
    for name, expected in summary_counts.items():
        if summary.get(name) is not None and int(summary[name]) != expected:
            raise ObserverError(f"scientific summary {name} does not match its source artifact")
    run_root = chronicle_root.parent if chronicle_root.name == "chronicle" else chronicle_root
    return _ScientificSource(
        run_root=run_root,
        chronicle_root=chronicle_root,
        manifest=manifest,
        summary=summary,
        events=events,
        entries=entries,
        episodes=episodes,
        hashes=hashes,
    )


def _rng_fingerprints() -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "python": stable_hash("observer-python-rng-state", repr(random.getstate())),
        "numpy": None,
        "torch": None,
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state = numpy.random.get_state()
        values["numpy"] = stable_hash(
            "observer-numpy-rng-state",
            [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        )
    torch = sys.modules.get("torch")
    if torch is not None:
        values["torch"] = stable_hash(
            "observer-torch-rng-state", torch.random.get_rng_state().tolist()
        )
    return values


def _template_choice(
    templates: Sequence[tuple[str, str]],
    *,
    source_id: str,
    audience_profile: str,
    template_family: str,
) -> tuple[str, str]:
    if not templates:
        raise ObserverError(f"template family is empty: {template_family}")
    digest = stable_hash(
        "eiel-observer-template-choice-v1",
        {
            "source_id": source_id,
            "audience_profile": audience_profile,
            "template_family": template_family,
        },
        length=16,
    )
    return templates[int(digest, 16) % len(templates)]


def validate_observer_wording(value: str) -> None:
    text = str(value or "")
    for pattern in FORBIDDEN_CLAIM_PATTERNS:
        if re.search(rf"\b(?:{pattern})\b", text, flags=re.IGNORECASE):
            raise ObserverError(f"Observer wording violates the claims contract: {pattern}")
    for pattern in RELIGIOUS_SPECIFICITY_PATTERNS:
        if re.search(rf"\b(?:{pattern})\b", text, flags=re.IGNORECASE):
            raise ObserverError(f"Observer wording violates the religious-specificity boundary: {pattern}")


def _measurement(event: Mapping[str, Any], key: str, default: str = "not available") -> str:
    value = dict(event.get("measurements", {}) or {}).get(key)
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _count_text(event: Mapping[str, Any], key: str, singular: str, plural: str | None = None) -> str:
    rendered = _measurement(event, key)
    noun = singular if rendered == "1" else (plural or f"{singular}s")
    return f"{rendered} {noun}"


def _lineage(event: Mapping[str, Any]) -> str:
    lineages = [str(value) for value in event.get("lineage_ids", []) or []]
    return lineages[0] if lineages else str(event.get("subject_id") or "an unidentified lineage")


def _location(event: Mapping[str, Any]) -> str:
    if event.get("region_id"):
        return f" in region {event['region_id']}"
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return f" at patch ({int(coordinates[0])}, {int(coordinates[1])})"
    return ""


def _fraction_percent(event: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = dict(event.get("measurements", {}) or {}).get(key)
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def _event_plain(event: Mapping[str, Any]) -> str:
    """Direct public wording. Scientific caveats remain in structured limitations."""
    event_type = str(event.get("event_type", "recorded_event"))
    generation = int(event.get("generation", 0))
    lineage = _lineage(event)
    location = _location(event)
    subject = str(event.get("subject_id") or "the recorded subject")
    if event_type == "generation_summary":
        return (
            f"Generation {generation} ended with {_count_text(event, 'alive', 'living agent')}, "
            f"{_count_text(event, 'births', 'birth')}, {_count_text(event, 'deaths', 'death')}, "
            f"and {_count_text(event, 'failed_births', 'failed birth')}."
        )
    if event_type in {"notable_birth", "birth_observation"}:
        agent = str((event.get("agent_ids") or [subject])[0])
        return f"Agent {agent} was born into lineage {lineage}{location}."
    if event_type in {"notable_death", "death_observation"}:
        agent = str((event.get("agent_ids") or [subject])[0])
        return f"Agent {agent} died in lineage {lineage}{location}."
    if event_type == "birth_failure_observation":
        return f"A birth attempt involving {subject}{location} ended without a birth."
    if event_type == "mutation_observation":
        return (
            f"A genome change was recorded for {subject}. Current evidence does not establish inheritance "
            "or an effect on later outcomes."
        )
    if event_type == "lineage_appearance":
        return (
            f"Lineage {lineage} entered the record with {_count_text(event, 'alive_count', 'living member')} "
            f"and population share {_measurement(event, 'alive_share')}."
        )
    if event_type == "lineage_dominance_crossing":
        return (
            f"Lineage {lineage} rose from share {_measurement(event, 'share_before')} to "
            f"{_measurement(event, 'share_after')}, crossing the dominance threshold "
            f"{_measurement(event, 'threshold')}."
        )
    if event_type == "lineage_leadership_change":
        return (
            f"The leading lineage changed at generation {generation}; the new leader held share "
            f"{_measurement(event, 'new_leader_share')}."
        )
    if event_type in {"lineage_rapid_share_increase", "lineage_rapid_share_decline"}:
        direction = "rose" if event_type.endswith("increase") else "fell"
        return (
            f"Lineage {lineage} {direction} from share {_measurement(event, 'share_before')} to "
            f"{_measurement(event, 'share_after')}."
        )
    if event_type == "lineage_near_extinction":
        return f"Lineage {lineage} now has {_count_text(event, 'alive_after', 'living member')} remaining."
    if event_type == "lineage_extinction":
        return f"Lineage {lineage} has no living representatives."
    if event_type == "branch_candidate":
        return (
            f"The source reporter flagged {subject} as a branch candidate with "
            f"{_measurement(event, 'candidate_alive_count')} living members. The label identifies a candidate, "
            "not a controlled branch result."
        )
    if event_type == "regional_food_depletion":
        return (
            f"Food scarcity spread from {_fraction_percent(event, 'depleted_fraction_before')} to "
            f"{_fraction_percent(event, 'depleted_fraction_after')} of recorded patches{location}."
        )
    if event_type == "regional_food_recovery":
        return (
            f"The food-depleted share fell from {_fraction_percent(event, 'depleted_fraction_before')} to "
            f"{_fraction_percent(event, 'depleted_fraction_after')}{location}."
        )
    if event_type == "crowding_hotspot":
        return (
            f"Crowding hotspots reached {_fraction_percent(event, 'hotspot_fraction_after')} of the recorded area, "
            f"against threshold {_fraction_percent(event, 'threshold')}{location}."
        )
    if event_type == "waste_hotspot":
        return (
            f"Mean waste on occupied patches reached {_measurement(event, 'occupied_waste_after')} "
            f"against threshold {_measurement(event, 'threshold')}{location}."
        )
    if event_type == "regrowth_suppression":
        return f"The environment recorded {_measurement(event, 'regrowth_suppressed_total')} total regrowth suppression."
    if event_type in {"predator_pressure_spike", "predator_damage"}:
        return (
            f"The record contains {_measurement(event, 'predator_events')} predator events and "
            f"{_measurement(event, 'predator_damage')} total predator damage{location}."
        )
    if event_type == "ecological_recovery":
        return (
            f"Positive ecological-recovery activity changed from {_measurement(event, 'previous_total')} "
            f"to {_measurement(event, 'positive_recovery_total')}{location}."
        )
    if event_type == "population_bottleneck":
        return (
            f"{_measurement(event, 'alive')} of {_measurement(event, 'cohort_size')} starting cohort members "
            f"remained alive, a survival fraction of {_measurement(event, 'survival_fraction')}."
        )
    if event_type == "delayed_hazard_cycle":
        return (
            f"The delayed-hazard cycle contained {_measurement(event, 'event_count')} events, "
            f"{_measurement(event, 'agents_damaged')} damaged agents, and "
            f"{_measurement(event, 'damage_total')} total damage."
        )
    measured = []
    for key, value in sorted(dict(event.get("measurements", {}) or {}).items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            measured.append(f"{key.replace('_', ' ')}={value}")
        if len(measured) >= 4:
            break
    suffix = "; ".join(measured) if measured else "no compact measurements were available"
    if str(event.get("event_category")) == "integrity" or event_type in _TECHNICAL_EVENT_TYPES:
        return f"A technical Chronicle limitation was recorded: {suffix}."
    return f"Chronicle recorded {event_type.replace('_', ' ')} at generation {generation}: {suffix}."


def _humour_family(event: Mapping[str, Any], episode_family: str | None = None) -> str:
    event_type = str(event.get("event_type", ""))
    if str(event.get("event_category")) == "integrity" or event_type in _TECHNICAL_EVENT_TYPES:
        return "integrity"
    if event_type == "generation_summary":
        return "generation_summary"
    if event_type == "birth_failure_observation":
        return "birth_failure"
    if event_type in {"notable_birth", "birth_observation"}:
        return "birth"
    if event_type in {"notable_death", "death_observation"}:
        return "death"
    if event_type == "mutation_observation":
        return "mutation"
    if event_type == "branch_candidate":
        return "branch_candidate"
    if event_type == "lineage_leadership_change":
        return "lineage_leadership"
    if event_type in {"lineage_appearance", "lineage_dominance_crossing", "lineage_rapid_share_increase"}:
        return "lineage_expansion"
    if event_type in {"lineage_near_extinction", "lineage_rapid_share_decline"}:
        return "lineage_decline"
    if event_type == "lineage_extinction":
        return "lineage_extinction_sequence"
    if event_type == "regional_food_depletion":
        return "regional_food_depletion"
    if event_type == "regional_food_recovery":
        return "food_recovery"
    if event_type in {"crowding_hotspot", "waste_hotspot"}:
        return "crowding_waste_crisis"
    if event_type == "regrowth_suppression":
        return "regrowth_suppression"
    if event_type == "predator_pressure_spike":
        return "predator_pressure_wave"
    if event_type == "predator_damage":
        return "predator_damage"
    if event_type == "ecological_recovery":
        return "ecological_recovery"
    if event_type in {"population_bottleneck", "delayed_hazard_cycle"}:
        return event_type
    if episode_family in DRY_HUMOUR_LINES:
        return str(episode_family)
    return "default"


def _number(event: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = dict(event.get("measurements", {}) or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _contextual_humour_candidates(event: Mapping[str, Any], humour_family: str) -> tuple[tuple[str, str], ...]:
    """Add deterministic run-specific angles without changing scientific meaning."""
    rows = list(DRY_HUMOUR_LINES.get(humour_family, DRY_HUMOUR_LINES["default"]))
    event_type = str(event.get("event_type", ""))
    if event_type == "generation_summary":
        births, deaths = _number(event, "births"), _number(event, "deaths")
        if births > deaths:
            rows.append(("obs2.context.summary.growth", "The population expanded. Capacity planning has been notified."))
        elif deaths > births:
            rows.append(("obs2.context.summary.decline", "The population contracted. The chart has saved space."))
        else:
            rows.append(("obs2.context.summary.balance", "Births and deaths reached an agreement. Nobody else was consulted."))
    elif event_type == "regional_food_depletion":
        after = _number(event, "depleted_fraction_after")
        if after >= 0.75:
            rows.append(("obs2.context.food.severe", "The menu is now mostly a historical document."))
        elif after >= 0.5:
            rows.append(("obs2.context.food.major", "More than half the menu has become unavailable in principle and practice."))
    elif event_type == "crowding_hotspot" and _number(event, "hotspot_fraction_after") >= 0.5:
        rows.append(("obs2.context.crowd.major", "Half the recorded area has joined the queue for personal space."))
    elif event_type in {"predator_pressure_spike", "predator_damage"}:
        if _number(event, "predator_events") >= 100:
            rows.append(("obs2.context.predator.busy", "This has progressed beyond an isolated incident and into scheduling."))
        if _number(event, "predator_damage") > 0:
            rows.append(("obs2.context.predator.damage", "The risk forecast has begun issuing receipts."))
    elif event_type == "population_bottleneck" and _number(event, "survival_fraction", 1.0) <= 0.2:
        rows.append(("obs2.context.bottleneck.severe", "The surviving cohort could now hold meetings without booking a large room."))
    elif event_type == "lineage_near_extinction" and _number(event, "alive_after") == 1:
        rows.append(("obs2.context.lineage.last", "The next family reunion requires one chair."))
    elif event_type == "lineage_extinction":
        rows.append(("obs2.context.lineage.extinct", "No further updates are expected from this branch of the family tree."))
    elif event_type == "delayed_hazard_cycle" and _number(event, "event_count") >= 10:
        rows.append(("obs2.context.hazard.repeated", "The delayed hazard repeated itself often enough to qualify as reliable."))
    elif event_type == "ecological_recovery" and _number(event, "positive_recovery_total") > _number(event, "previous_total"):
        rows.append(("obs2.context.recovery.rise", "Conditions improved. The Observer is withholding the parade pending further evidence."))
    return tuple(rows)


def _cosmic_contextual_humour_candidates(event: Mapping[str, Any], humour_family: str) -> tuple[tuple[str, str], ...]:
    """Add deterministic creator-profile reactions without changing source meaning."""
    rows = list(COSMIC_HUMOUR_LINES.get(humour_family, COSMIC_HUMOUR_LINES["default"]))
    event_type = str(event.get("event_type", ""))
    if event_type == "generation_summary":
        births, deaths = _number(event, "births"), _number(event, "deaths")
        if births > deaths:
            rows.append(("obs3.cosmic.context.summary.growth", "Creation expanded. The maintenance budget was not consulted."))
        elif deaths > births:
            rows.append(("obs3.cosmic.context.summary.decline", "Creation contracted. I will not call subtraction a plan."))
        else:
            rows.append(("obs3.cosmic.context.summary.balance", "Births and deaths balanced, which is not the same as peace."))
    elif event_type == "regional_food_depletion":
        after = _number(event, "depleted_fraction_after")
        if after >= 0.7:
            rows.append(("obs3.cosmic.context.food.severe", f"Food scarcity now covers {after * 100:.1f}% of recorded patches. I created abundance; evolution produced inventory management."))
        elif after >= 0.5:
            rows.append(("obs3.cosmic.context.food.major", "More than half the menu has left the world in spirit and practice."))
    elif event_type == "crowding_hotspot" and _number(event, "hotspot_fraction_after") >= 0.5:
        rows.append(("obs3.cosmic.context.crowd.major", "Half the recorded area has become a queue. I made more area than this."))
    elif event_type in {"predator_pressure_spike", "predator_damage"}:
        predator_events = _number(event, "predator_events")
        if predator_events >= 100:
            rows.append(("obs3.cosmic.context.predator.busy", f"The record contains {predator_events:.0f} predator events. The predators appear to have found catering."))
        if _number(event, "predator_damage") > 0:
            rows.append(("obs3.cosmic.context.predator.damage", "Risk produced receipts. Creation is reviewing the venue choice."))
    elif event_type == "population_bottleneck":
        fraction = _number(event, "survival_fraction", 1.0)
        if fraction <= 0.2:
            rows.append(("obs3.cosmic.context.bottleneck.severe", f"Only {fraction * 100:.1f}% of the starting cohort survived. The headcount is smaller, not better."))
    elif event_type == "lineage_near_extinction" and _number(event, "alive_after") == 1:
        rows.append(("obs3.cosmic.context.lineage.last", "One living representative remains. I recommend postponing the family reunion."))
    elif event_type == "lineage_extinction":
        rows.append(("obs3.cosmic.context.lineage.extinct", "No living representatives remain. Creation has simplified the file, regrettably."))
    elif event_type == "delayed_hazard_cycle" and _number(event, "event_count") >= 10:
        rows.append(("obs3.cosmic.context.hazard.repeated", "The delayed hazard repeated itself often enough to seem punctual."))
    elif event_type == "ecological_recovery" and _number(event, "positive_recovery_total") > _number(event, "previous_total"):
        rows.append(("obs3.cosmic.context.recovery.rise", "Conditions improved. I am prepared to call this progress, not competence."))
    return tuple(rows)


def _humour_candidates_for_profile(
    event: Mapping[str, Any],
    humour_family: str,
    *,
    audience_profile: str,
) -> tuple[tuple[str, str], ...]:
    if audience_profile == "cosmic_creator":
        return _cosmic_contextual_humour_candidates(event, humour_family)
    return _contextual_humour_candidates(event, humour_family)


def _profile_uses_humour(audience_profile: str) -> bool:
    return audience_profile in {"dry_humour", "cosmic_creator"}


def render_observer_event(
    event: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    audience_profile: str,
    episode_family: str | None = None,
    source_completeness: str,
) -> dict[str, Any]:
    if audience_profile not in AUDIENCE_PROFILES:
        raise ObserverError(f"unsupported audience profile: {audience_profile}")
    event_id = str(event["event_id"])
    generation = int(event.get("generation", 0))
    subject = str(event.get("subject_id") or "the recorded subject")
    event_type = str(event.get("event_type", "recorded_event"))
    plain = _event_plain(event)
    humour_family = _humour_family(event, episode_family)
    humour_id, humour = _template_choice(
        _humour_candidates_for_profile(event, humour_family, audience_profile=audience_profile),
        source_id=event_id,
        audience_profile=audience_profile,
        template_family=f"humour:{humour_family}",
    )
    frame_id, frame = _template_choice(
        PROFILE_FRAMES[audience_profile],
        source_id=event_id,
        audience_profile=audience_profile,
        template_family="event_frame",
    )
    text = frame.format(plain=plain, humour=humour, generation=generation)
    humour_line = humour if _profile_uses_humour(audience_profile) else None
    headline = EVENT_HEADLINES.get(event_type, "A Chronicle event was recorded").format(
        generation=generation,
        subject=subject,
    )
    for value in (headline, text, humour_line or ""):
        validate_observer_wording(value)
    observer_id = "obs_" + stable_hash(
        "eiel-observer-feed-item-v1",
        {"source_event_id": event_id, "audience_profile": audience_profile},
    )
    return canonical_payload(
        {
            "schema_version": OBSERVER_FEED_SCHEMA,
            "observer_item_id": observer_id,
            "audience_profile": audience_profile,
            "language": "en",
            "presentation_only": True,
            "generation": generation,
            "step": int(event.get("step", 0)),
            "event_type": event_type,
            "event_category": str(event.get("event_category", "")),
            "episode_family": episode_family,
            "technical_class": bool(
                str(event.get("event_category")) in _TECHNICAL_CATEGORIES
                or event_type in _TECHNICAL_EVENT_TYPES
            ),
            "headline": headline,
            "text": text,
            "plain_factual_text": plain,
            "humour_line": humour_line,
            "template_ids": {
                "headline": f"obs1.headline.event.{event_type}" if event_type in EVENT_HEADLINES else "obs1.headline.event.default",
                "frame": frame_id,
                "humour": humour_id if _profile_uses_humour(audience_profile) else None,
            },
            "presentation_dimensions": {
                "live_priority": str(event.get("live_priority", "background")),
                "severity": str(event.get("severity", "informational")),
                "scientific_importance": int(event.get("scientific_importance", 0)),
                "historical_importance": int(event.get("historical_importance", 0)),
                "public_interest": int(event.get("public_interest", 0)),
                "visual_interest": int(event.get("visual_interest", 0)),
            },
            "source_links": {
                "event_ids": [event_id],
                "entry_ids": [str(entry["chronicle_entry_id"])],
                "episode_ids": [str(entry["episode_id"])] if entry.get("episode_id") else [],
            },
            "source_provenance": {
                "run_id": str(event.get("run_id", "")),
                "campaign_id": event.get("campaign_id"),
                "branch_id": event.get("branch_id"),
                "checkpoint_segment": "source_record",
            },
            "subject_references": dict(entry.get("subject_references", {}) or {}),
            "confidence_class": str(event.get("confidence_class", "unknown")),
            "causality_class": str(event.get("causality_class", "not_assessed")),
            "source_completeness": source_completeness,
            "limitations": [
                "Observer content is deterministic presentation derived from the linked Scientific Chronicle record.",
                *list(entry.get("limitations", []) or []),
            ],
        }
    )


def _storage_event_eligible(event: Mapping[str, Any], profile: str) -> bool:
    event_type = str(event.get("event_type", ""))
    priority = str(event.get("live_priority", "background"))
    severity = str(event.get("severity", "informational"))
    public_interest = int(event.get("public_interest", 0))
    if profile == "observer_summary":
        return event_type == "generation_summary" or priority in {"critical", "high"} or severity in {
            "critical",
            "major",
        }
    if profile == "observer_notable":
        return (
            event_type == "generation_summary"
            or public_interest >= 1
            or _PRIORITY_RANK.get(priority, 4) <= _PRIORITY_RANK["high"]
            or severity in {"critical", "major"}
        )
    return True


def _subject_matches(event: Mapping[str, Any], config: ObserverConfig) -> bool:
    if config.agent_ids:
        observed = {int(value) for value in event.get("agent_ids", []) or []}
        if not observed.intersection(config.agent_ids):
            return False
    if config.lineage_ids:
        observed_lineages = {str(value) for value in event.get("lineage_ids", []) or []}
        if not observed_lineages.intersection(config.lineage_ids):
            return False
    if config.region_ids:
        region = str(event.get("region_id") or "")
        coordinates = event.get("patch_coordinates")
        patch = ""
        if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
            patch = f"patch:{int(coordinates[0])},{int(coordinates[1])}"
        if not {region, patch}.intersection(config.region_ids):
            return False
    return True


def _event_filter_eligible(event: Mapping[str, Any], config: ObserverConfig) -> bool:
    generation = int(event.get("generation", 0))
    if config.generation_start is not None and generation < int(config.generation_start):
        return False
    if config.generation_end is not None and generation > int(config.generation_end):
        return False
    if config.categories and str(event.get("event_category")) not in set(config.categories):
        return False
    if config.event_types and str(event.get("event_type")) not in set(config.event_types):
        return False
    if int(event.get("public_interest", 0)) < int(config.minimum_public_interest):
        return False
    if int(event.get("visual_interest", 0)) < int(config.minimum_visual_interest):
        return False
    priority = str(event.get("live_priority", "background"))
    if _PRIORITY_RANK.get(priority, 4) > _PRIORITY_RANK[config.minimum_live_priority]:
        return False
    return _subject_matches(event, config)


def _presentation_order_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _PRIORITY_RANK.get(str(event.get("live_priority", "background")), 4),
        -int(event.get("public_interest", 0)),
        -int(event.get("visual_interest", 0)),
        -int(event.get("historical_importance", 0)),
        -int(event.get("scientific_importance", 0)),
        event_order_key(event),
    )


def _theme_key(event: Mapping[str, Any]) -> tuple[str, str]:
    subject = str(event.get("subject_id") or "")
    if not subject:
        lineages = [str(value) for value in event.get("lineage_ids", []) or []]
        subject = lineages[0] if lineages else str(event.get("region_id") or "global")
    return str(event.get("event_type", "")), subject


def select_observer_events(
    source: _ScientificSource,
    config: ObserverConfig,
) -> list[tuple[dict[str, Any], dict[str, Any], str | None]]:
    config.validate()
    episodes_by_id = {str(episode["episode_id"]): episode for episode in source.episodes}
    candidates_by_generation: dict[int, list[tuple[dict[str, Any], dict[str, Any], str | None]]] = defaultdict(list)
    for event, entry in zip(source.events, source.entries):
        if not _storage_event_eligible(event, config.storage_profile):
            continue
        if not _event_filter_eligible(event, config):
            continue
        episode = episodes_by_id.get(str(entry.get("episode_id") or ""))
        family = str(episode.get("episode_type")) if episode else None
        if config.episode_families and family not in set(config.episode_families):
            continue
        candidates_by_generation[int(event.get("generation", 0))].append((event, entry, family))
    selected: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    last_theme_generation: dict[tuple[str, str], int] = {}
    for generation in sorted(candidates_by_generation):
        ranked = sorted(candidates_by_generation[generation], key=lambda row: _presentation_order_key(row[0]))
        retained: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
        for row in ranked:
            theme = _theme_key(row[0])
            last_generation = last_theme_generation.get(theme)
            if (
                last_generation is not None
                and generation - last_generation <= config.repeated_theme_cooldown
                and str(row[0].get("event_type")) != "generation_summary"
            ):
                continue
            retained.append(row)
            last_theme_generation[theme] = generation
            if len(retained) >= config.max_messages_per_generation:
                break
        selected.extend(sorted(retained, key=lambda row: event_order_key(row[0])))
        if len(selected) >= config.feed_budget:
            return selected[: config.feed_budget]
    return selected


def _episode_source_events(
    source: _ScientificSource,
    episode: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(event["event_id"]): event for event in source.events}
    source_ids = [str(value) for value in episode.get("source_event_ids", []) or []]
    return [by_id[event_id] for event_id in source_ids if event_id in by_id]


def _episode_dimensions(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "live_priority": "background",
            "scientific_importance": 0,
            "historical_importance": 0,
            "public_interest": 0,
            "visual_interest": 0,
        }
    return {
        "live_priority": min(
            (str(event.get("live_priority", "background")) for event in events),
            key=lambda value: _PRIORITY_RANK.get(value, 4),
        ),
        "scientific_importance": max(int(event.get("scientific_importance", 0)) for event in events),
        "historical_importance": max(int(event.get("historical_importance", 0)) for event in events),
        "public_interest": max(int(event.get("public_interest", 0)) for event in events),
        "visual_interest": max(int(event.get("visual_interest", 0)) for event in events),
    }


def _episode_plain(episode: Mapping[str, Any]) -> str:
    family = str(episode.get("episode_type", "recorded_episode"))
    start = int(episode.get("start_generation", 0))
    end = int(episode.get("end_generation", start))
    range_text = f"generation {start}" if start == end else f"generations {start} to {end}"
    lineages = [str(value) for value in episode.get("lineage_ids", []) or []]
    region = episode.get("affected_region")
    subject = f" for lineage {lineages[0]}" if lineages else f" in {region}" if region else ""
    descriptions = {
        "regional_food_depletion": "Food depletion remained active",
        "crowding_waste_crisis": "Crowding and waste pressure remained active",
        "predator_pressure_wave": "Predator pressure remained elevated",
        "ecological_recovery": "Ecological recovery remained active",
        "lineage_expansion": "Lineage expansion continued",
        "lineage_decline": "Lineage decline continued",
        "lineage_extinction_sequence": "The lineage moved through an extinction sequence",
        "population_bottleneck": "The population remained in a bottleneck",
        "delayed_hazard_cycle": "The delayed-hazard cycle remained active",
    }
    status = str(episode.get("status", "unknown"))
    status_text = "The episode is closed." if status == "closed" else "The episode remains open." if status == "open" else f"The episode status is {status}."
    return f"{descriptions.get(family, family.replace('_', ' ').capitalize())}{subject} across {range_text}. {status_text}"


def render_observer_episode(
    source: _ScientificSource,
    episode: Mapping[str, Any],
    *,
    audience_profile: str,
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    family = str(episode.get("episode_type", "recorded_episode"))
    start = int(episode.get("start_generation", 0))
    plain = _episode_plain(episode)
    humour_rows = (
        COSMIC_HUMOUR_LINES.get(family, COSMIC_HUMOUR_LINES["default"])
        if audience_profile == "cosmic_creator"
        else DRY_HUMOUR_LINES.get(family, DRY_HUMOUR_LINES["default"])
    )
    humour_id, humour = _template_choice(
        humour_rows,
        source_id=episode_id,
        audience_profile=audience_profile,
        template_family=f"episode_humour:{family}",
    )
    frame_id, frame = _template_choice(
        PROFILE_FRAMES[audience_profile],
        source_id=episode_id,
        audience_profile=audience_profile,
        template_family="episode_frame",
    )
    text = frame.format(plain=plain, humour=humour, generation=start)
    headline = EPISODE_HEADLINES.get(family, "A recorded Chronicle episode")
    for value in (headline, text, humour if _profile_uses_humour(audience_profile) else ""):
        validate_observer_wording(value)
    source_events = _episode_source_events(source, episode)
    dimensions = _episode_dimensions(source_events)
    return canonical_payload(
        {
            "schema_version": OBSERVER_EPISODE_SCHEMA,
            "observer_episode_id": "obsepi_" + stable_hash(
                "eiel-observer-episode-v1",
                {"source_episode_id": episode_id, "audience_profile": audience_profile},
            ),
            "audience_profile": audience_profile,
            "language": "en",
            "presentation_only": True,
            "episode_family": family,
            "headline": headline,
            "text": text,
            "plain_factual_text": plain,
            "humour_line": humour if _profile_uses_humour(audience_profile) else None,
            "template_ids": {
                "headline": f"obs1.headline.episode.{family}" if family in EPISODE_HEADLINES else "obs1.headline.episode.default",
                "frame": frame_id,
                "humour": humour_id if _profile_uses_humour(audience_profile) else None,
            },
            "start_generation": start,
            "end_generation": int(episode.get("end_generation", start)),
            "status": str(episode.get("status", "unknown")),
            "subject": {
                "region": episode.get("affected_region"),
                "agent_ids": list(episode.get("agent_ids", []) or []),
                "lineage_ids": list(episode.get("lineage_ids", []) or []),
            },
            "key_measurements": {
                "baseline": dict(episode.get("baseline_measurements", {}) or {}),
                "final": dict(episode.get("final_measurements", {}) or {}),
            },
            "presentation_dimensions": dimensions,
            "source_links": {
                "episode_id": episode_id,
                "event_ids": [str(value) for value in episode.get("source_event_ids", []) or []],
            },
            "confidence_class": str(episode.get("confidence_class", "unknown")),
            "causality_class": str(episode.get("causality_class", "not_assessed")),
            "source_completeness": str(
                source.manifest.get("source_completeness", source.summary.get("source_completeness", "unknown"))
            ),
            "limitations": [
                "Observer content is presentation of one frozen scientific episode; episodes were not collapsed.",
                *list(episode.get("limitations", []) or []),
            ],
        }
    )


def select_observer_episodes(
    source: _ScientificSource,
    config: ObserverConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in source.episodes:
        family = str(episode.get("episode_type", ""))
        if config.episode_families and family not in set(config.episode_families):
            continue
        start = int(episode.get("start_generation", 0))
        end = int(episode.get("end_generation", start))
        if config.generation_start is not None and end < int(config.generation_start):
            continue
        if config.generation_end is not None and start > int(config.generation_end):
            continue
        source_events = _episode_source_events(source, episode)
        if config.agent_ids and not set(config.agent_ids).intersection(
            int(value) for value in episode.get("agent_ids", []) or []
        ):
            continue
        if config.lineage_ids and not set(config.lineage_ids).intersection(
            str(value) for value in episode.get("lineage_ids", []) or []
        ):
            continue
        if config.region_ids and str(episode.get("affected_region") or "global") not in set(config.region_ids):
            continue
        if config.categories and not any(
            str(event.get("event_category")) in set(config.categories) for event in source_events
        ):
            continue
        if config.event_types and not any(
            str(event.get("event_type")) in set(config.event_types) for event in source_events
        ):
            continue
        dimensions = _episode_dimensions(source_events)
        if int(dimensions["public_interest"]) < config.minimum_public_interest:
            continue
        if int(dimensions["visual_interest"]) < config.minimum_visual_interest:
            continue
        if _PRIORITY_RANK.get(str(dimensions["live_priority"]), 4) > _PRIORITY_RANK[config.minimum_live_priority]:
            continue
        if config.storage_profile == "observer_summary" and not (
            str(episode.get("peak_severity")) in {"critical", "major"}
            or _PRIORITY_RANK.get(str(dimensions["live_priority"]), 4) <= _PRIORITY_RANK["high"]
        ):
            continue
        if config.storage_profile == "observer_notable" and not (
            int(dimensions["public_interest"]) >= 1
            or _PRIORITY_RANK.get(str(dimensions["live_priority"]), 4) <= _PRIORITY_RANK["medium"]
            or str(episode.get("peak_severity")) in {"critical", "major", "moderate"}
        ):
            continue
        rows.append(render_observer_episode(source, episode, audience_profile=config.audience_profile))
    return sorted(
        rows,
        key=lambda row: (
            int(row["start_generation"]),
            int(row["end_generation"]),
            str(row["source_links"]["episode_id"]),
        ),
    )


def _safe_gen1_payload(gen1_root: Path | None, run_id: str) -> dict[str, Any]:
    unavailable = {
        "status": "unavailable",
        "reason": "GEN1-R1 companion was not supplied for this Observer generation",
        "agent_mutations": {},
        "lineage_mutations": {},
        "event_mutations": {},
        "mutation_records": {},
        "source_hashes": [],
    }
    if gen1_root is None:
        return unavailable
    root = gen1_root.expanduser()
    if root.is_symlink():
        raise ObserverError("GEN1-R1 root must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise ObserverError("GEN1-R1 root is not a directory")
    companion_path = root / "chronicle_companion_index.json"
    inventory_path = root / "mutation_event_inventory.jsonl"
    companion = _load_json(companion_path, dict)
    if str(companion.get("schema_version")) != "eiel_gen1_chronicle_companion_index_v1":
        raise ObserverError("unsupported GEN1-R1 companion schema")
    inventory = _load_jsonl(inventory_path)
    mutation_records = {
        str(row.get("mutation_event_id")): row
        for row in inventory
        if str(row.get("run_id")) == run_id and str(row.get("evidence_strength")) == "observed"
    }
    run = dict(dict(companion.get("runs", {}) or {}).get(run_id, {}) or {})

    def filtered_map(name: str) -> dict[str, list[str]]:
        return {
            str(subject): sorted(
                mutation_id
                for mutation_id in {str(value) for value in values or []}
                if mutation_id in mutation_records
            )
            for subject, values in sorted(dict(run.get(name, {}) or {}).items())
        }

    event_map: dict[str, list[str]] = {}
    for name in ("chronicle_event_mutations", "birth_event_mutations"):
        for event_id, values in filtered_map(name).items():
            event_map.setdefault(event_id, [])
            event_map[event_id] = sorted(set(event_map[event_id]).union(values))
    return canonical_payload(
        {
            "status": "available" if run else "unavailable",
            "reason": None if run else "GEN1-R1 companion contains no row for the selected Chronicle run",
            "agent_mutations": filtered_map("agent_mutations"),
            "lineage_mutations": filtered_map("lineage_mutations"),
            "event_mutations": event_map,
            "mutation_records": mutation_records,
            "source_hashes": [
                {"relative_path": companion_path.name, "sha256": sha256_file(companion_path), "bytes": companion_path.stat().st_size},
                {"relative_path": inventory_path.name, "sha256": sha256_file(inventory_path), "bytes": inventory_path.stat().st_size},
            ],
        }
    )


def _card_summary(
    *,
    audience_profile: str,
    kind: str,
    identifier: str,
    event_count: int,
    first_generation: int,
    last_generation: int,
) -> tuple[str, str | None]:
    label = kind.capitalize()
    range_text = (
        f"in generation {first_generation}"
        if first_generation == last_generation
        else f"across generations {first_generation} to {last_generation}"
    )
    record_noun = "record" if event_count == 1 else "records"
    plain = (
        f"{label} {identifier} appears in {event_count} selected Observer {record_noun} {range_text}. "
        "This card includes only events preserved in the available Chronicle."
    )
    humour_template_id: str | None = None
    if audience_profile == "field_observer":
        result = f"Field record: {plain}"
    elif audience_profile == "dry_humour":
        humour_template_id, humour = _template_choice(
            CARD_HUMOUR_LINES[kind],
            source_id=f"{kind}:{identifier}",
            audience_profile=audience_profile,
            template_family=f"{kind}_story_card_humour",
        )
        result = f"{plain} {humour}"
    elif audience_profile == "cosmic_creator":
        humour_template_id, humour = _template_choice(
            COSMIC_CARD_HUMOUR_LINES[kind],
            source_id=f"{kind}:{identifier}",
            audience_profile=audience_profile,
            template_family=f"{kind}_story_card_humour",
        )
        result = f"{plain} {humour}"
    elif audience_profile == "documentary":
        result = f"{plain} This public account remains linked to the listed scientific records."
    else:
        result = plain
    validate_observer_wording(result)
    return result, humour_template_id


def _mutation_occurrences(
    mutation_ids: Iterable[str],
    gen1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = dict(gen1.get("mutation_records", {}) or {})
    rows: list[dict[str, Any]] = []
    for mutation_id in sorted({str(value) for value in mutation_ids}):
        record = dict(records.get(mutation_id, {}) or {})
        if str(record.get("evidence_strength")) != "observed":
            continue
        causal_status = str(record.get("causal_status", "not_assessed"))
        if causal_status not in {"not_assessed", "association_only"}:
            raise ObserverError(f"unsupported GEN1-R1 causal status: {causal_status}")
        rows.append(
            {
                "mutation_event_id": mutation_id,
                "changed_fields": list(record.get("changed_fields", []) or []),
                "evidence_strength": "observed",
                "source_completeness": str(record.get("source_completeness", "unknown")),
                "public_statement": (
                    "A genome change was recorded. Current evidence does not show whether descendants "
                    "inherited it or whether it affected an outcome."
                ),
                "inheritance_status": "unavailable",
                "persistence_status": "unavailable",
                "association_status": "unavailable",
                "causal_status": causal_status,
            }
        )
    return rows


def _episode_ids_by_event(observer_episodes: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for episode in observer_episodes:
        episode_id = str(dict(episode.get("source_links", {}) or {}).get("episode_id", ""))
        for event_id in dict(episode.get("source_links", {}) or {}).get("event_ids", []) or []:
            result[str(event_id)].append(episode_id)
    return {key: sorted(set(values)) for key, values in sorted(result.items())}


def _build_agent_cards(
    feed: Sequence[Mapping[str, Any]],
    observer_episodes: Sequence[Mapping[str, Any]],
    *,
    audience_profile: str,
    source_completeness: str,
    gen1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feed:
        for agent_id in dict(row.get("subject_references", {}) or {}).get("agent_ids", []) or []:
            grouped[str(agent_id)].append(row)
    episode_by_event = _episode_ids_by_event(observer_episodes)
    mutation_map = dict(gen1.get("agent_mutations", {}) or {})
    cards: list[dict[str, Any]] = []
    for agent_id, rows in sorted(grouped.items(), key=lambda pair: pair[0]):
        source_event_ids = sorted(
            {
                str(event_id)
                for row in rows
                for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
            }
        )
        generations = [int(row.get("generation", 0)) for row in rows]
        lineages = sorted(
            {
                str(value)
                for row in rows
                for value in dict(row.get("subject_references", {}) or {}).get("lineage_ids", []) or []
            }
        )
        episode_ids = sorted(
            {
                episode_id
                for event_id in source_event_ids
                for episode_id in episode_by_event.get(event_id, [])
            }
        )
        mutations = _mutation_occurrences(mutation_map.get(agent_id, []), gen1)
        event_types = sorted({str(row.get("event_type")) for row in rows})
        birth_event_ids = sorted(
            str(event_id)
            for row in rows
            if str(row.get("event_type")) in {"notable_birth", "birth_observation"}
            for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
        )
        death_event_ids = sorted(
            str(event_id)
            for row in rows
            if str(row.get("event_type")) in {"notable_death", "death_observation"}
            for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
        )
        summary, humour_template_id = _card_summary(
            audience_profile=audience_profile,
            kind="agent",
            identifier=agent_id,
            event_count=len(rows),
            first_generation=min(generations),
            last_generation=max(generations),
        )
        cards.append(
            canonical_payload(
                {
                    "schema_version": OBSERVER_AGENT_SCHEMA,
                    "observer_card_id": "obsagent_" + stable_hash(
                        "eiel-observer-agent-card-v1",
                        {"agent_id": agent_id, "audience_profile": audience_profile, "events": source_event_ids},
                    ),
                    "audience_profile": audience_profile,
                    "presentation_only": True,
                    "agent_id": agent_id,
                    "lineage_ids": lineages,
                    "summary": summary,
                    "template_ids": {"dry_humour": humour_template_id} if humour_template_id else {},
                    "generation_range": [min(generations), max(generations)],
                    "notable_event_count": len(rows),
                    "notable_event_types": event_types,
                    "recorded_birth_event_ids": birth_event_ids,
                    "recorded_death_event_ids": death_event_ids,
                    "source_event_ids": source_event_ids,
                    "source_episode_ids": episode_ids,
                    "mutation_occurrences": mutations,
                    "source_completeness": source_completeness,
                    "complete_biography": False,
                    "notability_rule": "referenced_by_selected_observer_feed",
                    "limitations": [
                        "This card is a bounded index of selected Chronicle records, not a complete biography.",
                        *(
                            ["GEN1-R1 mutation links remain occurrence-only; inheritance and effect are unavailable."]
                            if mutations
                            else []
                        ),
                    ],
                }
            )
        )
    return cards


def _build_lineage_cards(
    feed: Sequence[Mapping[str, Any]],
    observer_episodes: Sequence[Mapping[str, Any]],
    *,
    audience_profile: str,
    source_completeness: str,
    gen1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feed:
        for lineage_id in dict(row.get("subject_references", {}) or {}).get("lineage_ids", []) or []:
            grouped[str(lineage_id)].append(row)
    mutation_map = dict(gen1.get("lineage_mutations", {}) or {})
    cards: list[dict[str, Any]] = []
    for lineage_id, rows in sorted(grouped.items()):
        source_event_ids = sorted(
            {
                str(event_id)
                for row in rows
                for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
            }
        )
        episodes = [
            episode
            for episode in observer_episodes
            if lineage_id in {str(value) for value in dict(episode.get("subject", {}) or {}).get("lineage_ids", []) or []}
        ]
        generations = [int(row.get("generation", 0)) for row in rows]
        event_types = sorted({str(row.get("event_type")) for row in rows})
        regions = sorted(
            {
                str(dict(row.get("subject_references", {}) or {}).get("region_id"))
                for row in rows
                if dict(row.get("subject_references", {}) or {}).get("region_id")
            }
        )
        mutations = _mutation_occurrences(mutation_map.get(lineage_id, []), gen1)
        event_ids_by_type = {
            event_type: sorted(
                str(event_id)
                for row in rows
                if str(row.get("event_type")) == event_type
                for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
            )
            for event_type in (
                "lineage_appearance",
                "lineage_rapid_share_increase",
                "lineage_dominance_crossing",
                "lineage_rapid_share_decline",
                "lineage_near_extinction",
                "lineage_extinction",
                "lineage_leadership_change",
            )
        }
        summary, humour_template_id = _card_summary(
            audience_profile=audience_profile,
            kind="lineage",
            identifier=lineage_id,
            event_count=len(rows),
            first_generation=min(generations),
            last_generation=max(generations),
        )
        cards.append(
            canonical_payload(
                {
                    "schema_version": OBSERVER_LINEAGE_SCHEMA,
                    "observer_card_id": "obslineage_" + stable_hash(
                        "eiel-observer-lineage-card-v1",
                        {"lineage_id": lineage_id, "audience_profile": audience_profile, "events": source_event_ids},
                    ),
                    "audience_profile": audience_profile,
                    "presentation_only": True,
                    "lineage_id": lineage_id,
                    "summary": summary,
                    "template_ids": {"dry_humour": humour_template_id} if humour_template_id else {},
                    "generation_range": [min(generations), max(generations)],
                    "recorded_event_types": event_types,
                    "first_appearance_event_ids": event_ids_by_type["lineage_appearance"],
                    "expansion_event_ids": event_ids_by_type["lineage_rapid_share_increase"],
                    "dominance_event_ids": event_ids_by_type["lineage_dominance_crossing"],
                    "decline_event_ids": event_ids_by_type["lineage_rapid_share_decline"],
                    "near_extinction_event_ids": event_ids_by_type["lineage_near_extinction"],
                    "extinction_event_ids": event_ids_by_type["lineage_extinction"],
                    "leadership_event_ids": event_ids_by_type["lineage_leadership_change"],
                    "regions": regions,
                    "source_event_ids": source_event_ids,
                    "source_episode_ids": sorted(
                        str(dict(episode.get("source_links", {}) or {}).get("episode_id")) for episode in episodes
                    ),
                    "branch_candidate_status": "recorded_candidate_only" if "branch_candidate" in event_types else "not_recorded",
                    "mutation_occurrences": mutations,
                    "source_completeness": source_completeness,
                    "notability_rule": "referenced_by_selected_observer_feed",
                    "limitations": [
                        "This card reports recorded changes and co-occurrence; it does not assign lineage intent or a single cause.",
                        *(
                            ["GEN1-R1 mutation links remain occurrence-only; inheritance and effect are unavailable."]
                            if mutations
                            else []
                        ),
                    ],
                }
            )
        )
    return cards


def _event_region_keys(row: Mapping[str, Any]) -> list[str]:
    references = dict(row.get("subject_references", {}) or {})
    if references.get("region_id"):
        return [str(references["region_id"])]
    coordinates = references.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return [f"patch:{int(coordinates[0])},{int(coordinates[1])}"]
    return []


def _build_region_cards(
    feed: Sequence[Mapping[str, Any]],
    observer_episodes: Sequence[Mapping[str, Any]],
    *,
    audience_profile: str,
    source_completeness: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feed:
        for region_id in _event_region_keys(row):
            grouped[region_id].append(row)
    for episode in observer_episodes:
        region = dict(episode.get("subject", {}) or {}).get("region") or "global"
        grouped.setdefault(str(region), [])
    cards: list[dict[str, Any]] = []
    for region_id, rows in sorted(grouped.items()):
        episodes = [
            episode
            for episode in observer_episodes
            if str(dict(episode.get("subject", {}) or {}).get("region") or "global") == region_id
        ]
        direct_source_event_ids = sorted(
            {
                str(event_id)
                for row in rows
                for event_id in dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
            }
        )
        episode_source_event_ids = sorted(
            {
                str(event_id)
                for episode in episodes
                for event_id in dict(episode.get("source_links", {}) or {}).get("event_ids", []) or []
            }
        )
        source_event_ids = sorted(set(direct_source_event_ids).union(episode_source_event_ids))
        generations = [int(row.get("generation", 0)) for row in rows]
        generations.extend(int(episode.get("start_generation", 0)) for episode in episodes)
        generations.extend(int(episode.get("end_generation", 0)) for episode in episodes)
        if not generations:
            continue
        event_noun = "record" if len(rows) == 1 else "records"
        episode_noun = "story" if len(episodes) == 1 else "stories"
        range_text = (
            f"in generation {min(generations)}"
            if min(generations) == max(generations)
            else f"across generations {min(generations)} to {max(generations)}"
        )
        plain = (
            f"Region {region_id} has {len(rows)} directly selected event {event_noun} and "
            f"{len(episodes)} episode {episode_noun} {range_text}. The episode stories contribute "
            f"{len(episode_source_event_ids)} linked scientific "
            f"{'event' if len(episode_source_event_ids) == 1 else 'events'}."
        )
        field_text = f"Field record: {plain}"
        region_humour_rows = (
            COSMIC_CARD_HUMOUR_LINES["region"]
            if audience_profile == "cosmic_creator"
            else CARD_HUMOUR_LINES["region"]
        )
        humour_template_id, humour = _template_choice(
            region_humour_rows,
            source_id=f"region:{region_id}",
            audience_profile=audience_profile,
            template_family="region_card_humour",
        )
        selected_summary = {
            "public_plain": plain,
            "field_observer": field_text,
            "dry_humour": f"{plain} {humour}",
            "documentary": f"{plain} The account remains limited to linked Chronicle records.",
            "cosmic_creator": f"{plain} {humour}",
        }[audience_profile]
        for value in (plain, field_text, humour, selected_summary):
            validate_observer_wording(value)
        cards.append(
            canonical_payload(
                {
                    "schema_version": OBSERVER_REGION_SCHEMA,
                    "observer_card_id": "obsregion_" + stable_hash(
                        "eiel-observer-region-card-v1",
                        {"region_id": region_id, "audience_profile": audience_profile, "events": source_event_ids},
                    ),
                    "audience_profile": audience_profile,
                    "presentation_only": True,
                    "region_id": region_id,
                    "summary": selected_summary,
                    "plain_summary": plain,
                    "field_observer_summary": field_text,
                    "dry_humour_line": humour,
                    "template_ids": {"dry_humour": humour_template_id},
                    "generation_range": [min(generations), max(generations)],
                    "event_types": sorted({str(row.get("event_type")) for row in rows}),
                    "episode_families": sorted({str(episode.get("episode_family")) for episode in episodes}),
                    "lineage_ids": sorted(
                        {
                            str(value)
                            for row in rows
                            for value in dict(row.get("subject_references", {}) or {}).get("lineage_ids", []) or []
                        }
                    ),
                    "agent_ids": sorted(
                        {
                            str(value)
                            for row in rows
                            for value in dict(row.get("subject_references", {}) or {}).get("agent_ids", []) or []
                        }
                    ),
                    "source_event_ids": source_event_ids,
                    "direct_source_event_ids": direct_source_event_ids,
                    "episode_source_event_ids": episode_source_event_ids,
                    "source_episode_ids": sorted(
                        str(dict(episode.get("source_links", {}) or {}).get("episode_id")) for episode in episodes
                    ),
                    "source_completeness": source_completeness,
                    "limitations": [
                        "This region card summarizes selected event and episode membership without adding a causal explanation."
                    ],
                }
            )
        )
    return cards


def observer_storage_preflight(
    *,
    source_entry_count: int,
    source_episode_count: int,
    selected_entry_count: int,
    selected_episode_count: int,
    estimated_card_count: int,
    config: ObserverConfig,
) -> dict[str, Any]:
    config.validate()
    estimated_records = selected_entry_count + selected_episode_count + estimated_card_count + 4
    estimated_bytes = (
        selected_entry_count * 1800
        + selected_episode_count * 2400
        + estimated_card_count * 1600
        + 32_000
    )
    risk = "large" if config.storage_profile == "observer_full" else "compact"
    acknowledgements = ["acknowledge_large_output=true"] if config.storage_profile == "observer_full" else []
    return canonical_payload(
        {
            "schema_version": OBSERVER_PREFLIGHT_SCHEMA,
            "storage_profile": config.storage_profile,
            "source_entry_count": int(source_entry_count),
            "source_episode_count": int(source_episode_count),
            "selected_scientific_entries": int(selected_entry_count),
            "selected_scientific_episodes": int(selected_episode_count),
            "expected_observer_records": int(estimated_records),
            "estimated_bytes": int(estimated_bytes),
            "estimated_mebibytes": round(estimated_bytes / float(1024 * 1024), 6),
            "risk_classification": risk,
            "required_acknowledgements": acknowledgements,
            "accepted": not acknowledgements or config.acknowledge_large_output,
            "scientific_storage_accounting_affected": False,
        }
    )


def _source_limitations(source: _ScientificSource) -> list[str]:
    limitations = list(source.summary.get("limitations", []) or [])
    native_aggregate_only = False
    for event in source.events:
        measurements = dict(event.get("measurements", {}) or {})
        if measurements.get("native_lifecycle_authority") is False or (
            "native_typed_lifecycle_rows" in measurements
            and int(measurements.get("native_typed_lifecycle_rows", 0) or 0) == 0
        ):
            native_aggregate_only = True
            break
    if native_aggregate_only:
        limitations.append(
            "Native lifecycle coverage is aggregate-only; complete agent and lineage lifecycle judgments are unavailable."
        )
    return list(dict.fromkeys(str(value) for value in limitations if str(value)))


def _world_summary(
    source: _ScientificSource,
    feed: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    config: ObserverConfig,
) -> dict[str, Any]:
    generations = [int(row.get("generation", 0)) for row in feed]
    return canonical_payload(
        {
            "schema_version": OBSERVER_WORLD_SCHEMA,
            "run_id": str(source.manifest.get("run_id", "unknown")),
            "parent_checkpoint_id": source.manifest.get("parent_checkpoint_id"),
            "source_run_ids": sorted({str(event.get("run_id")) for event in source.events}),
            "source_branch_ids": sorted({str(event.get("branch_id") or "") for event in source.events}),
            "audience_profile": config.audience_profile,
            "storage_profile": config.storage_profile,
            "language": "en",
            "presentation_only": True,
            "headline": f"Observer history for run {source.manifest.get('run_id', 'unknown')}",
            "generation_range": [min(generations), max(generations)] if generations else None,
            "selected_event_count": len(feed),
            "episode_story_count": len(episodes),
            "agent_card_count": len(agents),
            "lineage_card_count": len(lineages),
            "region_card_count": len(regions),
            "counts_by_category": dict(sorted(Counter(str(row.get("event_category")) for row in feed).items())),
            "counts_by_live_priority": dict(
                sorted(
                    Counter(
                        str(dict(row.get("presentation_dimensions", {}) or {}).get("live_priority", "background"))
                        for row in feed
                    ).items()
                )
            ),
            "episode_counts_by_family": dict(
                sorted(Counter(str(row.get("episode_family")) for row in episodes).items())
            ),
            "source_completeness": str(
                source.manifest.get("source_completeness", source.summary.get("source_completeness", "unknown"))
            ),
            "limitations": [
                "This is deterministic public presentation, not a second scientific interpretation.",
                "Every public record links to frozen Scientific Chronicle source IDs.",
                *_source_limitations(source),
            ],
        }
    )


def _world_history_markdown(
    world: Mapping[str, Any],
    feed: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# {world['headline']}",
        "",
        f"Audience profile: `{world['audience_profile']}`",
        "",
        "Observer content is presentation derived from the linked Scientific Chronicle. The scientific source remains authoritative.",
        "",
        "## Episode History",
        "",
    ]
    if episodes:
        for episode in episodes:
            source_id = dict(episode.get("source_links", {}) or {}).get("episode_id", "unavailable")
            lines.extend(
                [
                    f"### {episode['headline']}",
                    "",
                    str(episode["text"]),
                    "",
                    f"Source episode: `{source_id}`",
                    "",
                ]
            )
    else:
        lines.extend(["No eligible scientific episodes were present in this Observer selection.", ""])
    lines.extend(["## Public Feed", ""])
    for row in feed:
        event_ids = dict(row.get("source_links", {}) or {}).get("event_ids", []) or []
        lines.append(
            f"- Generation {row['generation']}: {row['text']} "
            f"(source `{event_ids[0] if event_ids else 'unavailable'}`)"
        )
    if not feed:
        lines.append("No eligible public feed records were selected.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in world.get("limitations", []) or [])
    return "\n".join(lines) + "\n"


def _bounded_spoken_lines(text: str, maximum: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > maximum:
            if current:
                lines.append(" ".join(current))
                current = []
            lines.extend(word[index : index + maximum] for index in range(0, len(word), maximum))
            continue
        candidate = " ".join([*current, word])
        if current and len(candidate) > maximum:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _audio_script(
    feed: Sequence[Mapping[str, Any]],
    config: ObserverConfig,
    *,
    limitations: Sequence[str] = (),
) -> str:
    lines = [
        "EIEL Observer Chronicle audio script",
        f"Audience profile: {config.audience_profile}",
        "Presentation only. The linked Scientific Chronicle remains authoritative.",
        "",
    ]
    for limitation in limitations:
        lines.extend(_bounded_spoken_lines(f"Coverage note: {limitation}", config.maximum_audio_sentence_length))
    if limitations:
        lines.append("")
    previous_generation: int | None = None
    for row in feed:
        generation = int(row.get("generation", 0))
        if generation != previous_generation:
            lines.extend([f"[Generation {generation}]", ""])
            previous_generation = generation
        priority = str(dict(row.get("presentation_dimensions", {}) or {}).get("live_priority", "background"))
        prefix = f"[{priority.upper()}] " if priority in {"critical", "high"} else ""
        spoken = _bounded_spoken_lines(
            str(row.get("text", "")),
            config.maximum_audio_sentence_length - len(prefix),
        )
        lines.append(prefix + spoken[0])
        lines.extend(spoken[1:])
        lines.append("")
    if not feed:
        lines.append("No eligible Observer records were selected.")
    return "\n".join(lines).rstrip() + "\n"


def _artifact_inventory(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical_name, filename in sorted(OBSERVER_ARTIFACTS.items()):
        if logical_name == "manifest":
            continue
        path = output / filename
        if not path.is_file() or path.is_symlink():
            raise ObserverError(f"Observer output is missing: {filename}")
        rows.append(
            {
                "artifact": logical_name,
                "relative_path": filename,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return canonical_payload(rows)


def _require_reserved_output_path_available(path: Path) -> None:
    if path.is_symlink():
        raise ObserverError(f"reserved Observer publication path must not be a symlink: {path}")
    if path.exists():
        raise ObserverError(f"reserved Observer publication path already exists: {path}")


def _publish_observer_tree(staging: Path, output: Path) -> None:
    backup = output.with_name(f"{output.name}.obs1-previous")
    _require_reserved_output_path_available(backup)
    previous_moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            previous_moved = True
        os.replace(staging, output)
    except Exception:
        if previous_moved and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _observer_index(
    source: _ScientificSource,
    config: ObserverConfig,
    *,
    feed: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return canonical_payload(
        {
            "schema_version": OBSERVER_INDEX_SCHEMA,
            "run_id": str(source.manifest.get("run_id", "unknown")),
            "audience_profile": config.audience_profile,
            "presentation_only": True,
            "authoritative_source": "Scientific Chronicle",
            "feeds": dict(sorted(OBSERVER_ARTIFACTS.items())),
            "counts": {
                "feed": len(feed),
                "episodes": len(episodes),
                "agents": len(agents),
                "lineages": len(lineages),
                "regions": len(regions),
            },
            "filter_fields": [
                "audience_profile",
                "public_interest",
                "visual_interest",
                "live_priority",
                "category",
                "event_type",
                "episode_family",
                "generation",
                "agent_id",
                "lineage_id",
                "region_id",
            ],
            "source_link_fields": ["event_ids", "entry_ids", "episode_ids"],
            "observer_content_may_enter_scientific_analytics": False,
        }
    )


def write_observer_outputs(
    source_root: Path,
    output_root: Path,
    *,
    config: ObserverConfig,
    gen1_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a deterministic Observer package without mutating scientific inputs."""

    config.validate()
    rng_before = _rng_fingerprints()
    source = load_scientific_source(source_root)
    output = _safe_output_root(output_root, source)
    source_hashes_before = source.hashes
    run_id = str(source.manifest.get("run_id", "unknown"))
    source_completeness = str(
        source.manifest.get("source_completeness", source.summary.get("source_completeness", "unknown"))
    )
    gen1 = _safe_gen1_payload(gen1_root, run_id)
    selected = select_observer_events(source, config)
    observer_feed: list[dict[str, Any]] = []
    for event, entry, episode_family in selected:
        row = render_observer_event(
            event,
            entry,
            audience_profile=config.audience_profile,
            episode_family=episode_family,
            source_completeness=source_completeness,
        )
        row = dict(row)
        row["source_provenance"] = {
            **dict(row.get("source_provenance", {}) or {}),
            "checkpoint_segment": (
                "current_run"
                if str(event.get("run_id")) == run_id
                else "inherited_checkpoint_prefix"
            ),
        }
        mutation_ids = list(dict(gen1.get("event_mutations", {}) or {}).get(str(event["event_id"]), []) or [])
        mutations = _mutation_occurrences(mutation_ids, gen1)
        if mutations:
            causal_statuses = sorted({str(mutation["causal_status"]) for mutation in mutations})
            row["gen1_r1"] = {
                "status": "observed_only",
                "mutation_occurrences": mutations,
                "inheritance_status": "unavailable",
                "persistence_status": "unavailable",
                "association_status": "unavailable",
                "causal_status": causal_statuses[0] if len(causal_statuses) == 1 else "not_assessed",
                "source_causal_statuses": causal_statuses,
            }
        observer_feed.append(canonical_payload(row))
    observer_episodes = select_observer_episodes(source, config)
    agent_cards = _build_agent_cards(
        observer_feed,
        observer_episodes,
        audience_profile=config.audience_profile,
        source_completeness=source_completeness,
        gen1=gen1,
    )
    lineage_cards = _build_lineage_cards(
        observer_feed,
        observer_episodes,
        audience_profile=config.audience_profile,
        source_completeness=source_completeness,
        gen1=gen1,
    )
    region_cards = _build_region_cards(
        observer_feed,
        observer_episodes,
        audience_profile=config.audience_profile,
        source_completeness=source_completeness,
    )
    preflight = observer_storage_preflight(
        source_entry_count=len(source.entries),
        source_episode_count=len(source.episodes),
        selected_entry_count=len(observer_feed),
        selected_episode_count=len(observer_episodes),
        estimated_card_count=len(agent_cards) + len(lineage_cards) + len(region_cards),
        config=config,
    )
    if not preflight["accepted"]:
        raise ObserverError("Observer storage preflight was not accepted")
    world = _world_summary(
        source,
        observer_feed,
        observer_episodes,
        agent_cards,
        lineage_cards,
        region_cards,
        config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.obs1-staging")
    _require_reserved_output_path_available(staging)
    staging.mkdir()
    try:
        write_jsonl(staging / OBSERVER_ARTIFACTS["feed"], observer_feed)
        write_jsonl(staging / OBSERVER_ARTIFACTS["episodes"], observer_episodes)
        write_jsonl(staging / OBSERVER_ARTIFACTS["agents"], agent_cards)
        write_jsonl(staging / OBSERVER_ARTIFACTS["lineages"], lineage_cards)
        write_jsonl(staging / OBSERVER_ARTIFACTS["regions"], region_cards)
        write_json(staging / OBSERVER_ARTIFACTS["world_summary"], world)
        atomic_write_text(
            staging / OBSERVER_ARTIFACTS["world_history"],
            _world_history_markdown(world, observer_feed, observer_episodes),
        )
        atomic_write_text(
            staging / OBSERVER_ARTIFACTS["audio_script"],
            _audio_script(observer_feed, config, limitations=_source_limitations(source)),
        )
        write_json(staging / OBSERVER_ARTIFACTS["templates"], template_inventory())
        write_json(staging / OBSERVER_ARTIFACTS["preflight"], preflight)
        write_json(
            staging / OBSERVER_ARTIFACTS["index"],
            _observer_index(
                source,
                config,
                feed=observer_feed,
                episodes=observer_episodes,
                agents=agent_cards,
                lineages=lineage_cards,
                regions=region_cards,
            ),
        )
        source_hashes_after = _scientific_hashes(source.chronicle_root)
        rng_after = _rng_fingerprints()
        if source_hashes_before != source_hashes_after:
            raise ObserverError("scientific source bytes changed during Observer generation")
        if rng_before != rng_after:
            raise ObserverError("RNG state changed during Observer generation")
        artifacts = _artifact_inventory(staging)
        manifest = canonical_payload(
            {
                "schema_version": OBSERVER_MANIFEST_SCHEMA,
                "observer_version": OBSERVER_VERSION,
                "run_id": run_id,
                "campaign_id": source.manifest.get("campaign_id"),
                "branch_id": source.manifest.get("branch_id"),
                "parent_checkpoint_id": source.manifest.get("parent_checkpoint_id"),
                "source_commit": source.manifest.get("source_commit"),
                "audience_profile": config.audience_profile,
                "storage_profile": config.storage_profile,
                "language": "en",
                "deterministic_generation_mode": "stable_source_id_template_selection_v1",
                "presentation_only": True,
                "authoritative_source": "Scientific Chronicle",
                "source_chronicle_root": str(source.chronicle_root),
                "source_chronicle_artifacts": source_hashes_before,
                "source_completeness": source_completeness,
                "source_event_count": len(source.events),
                "source_run_ids": sorted({str(event.get("run_id")) for event in source.events}),
                "source_branch_ids": sorted({str(event.get("branch_id") or "") for event in source.events}),
                "source_entry_count": len(source.entries),
                "source_episode_count": len(source.episodes),
                "observer_feed_count": len(observer_feed),
                "observer_episode_count": len(observer_episodes),
                "observer_agent_count": len(agent_cards),
                "observer_lineage_count": len(lineage_cards),
                "observer_region_count": len(region_cards),
                "config": config.to_dict(),
                "presentation_order": [
                    "live_priority",
                    "public_interest_desc",
                    "visual_interest_desc",
                    "historical_importance_desc",
                    "scientific_importance_desc",
                    "formal_event_order",
                ],
                "independent_scientific_dimensions_preserved": True,
                "gen1_r1": {
                    "status": gen1.get("status"),
                    "reason": gen1.get("reason"),
                    "source_hashes": gen1.get("source_hashes", []),
                    "maximum_real_evidence_strength": "observed",
                    "inheritance_claims": "unavailable",
                    "persistence_claims": "unavailable",
                    "association_claims": "unavailable",
                    "causal_claims": "not_assessed",
                },
                "scientific_source_immutability": True,
                "scientific_ids_unchanged": True,
                "rng_immutability": True,
                "wall_clock_fields_emitted": False,
                "network_services_used": False,
                "simulation_launched": False,
                "observer_imported_into_scientific_analytics": False,
                "humour_contract": HUMOUR_CONTRACT["schema_version"],
                "artifacts": artifacts,
                "output_bytes_excluding_manifest": sum(int(row["bytes"]) for row in artifacts),
                "limitations": list(world.get("limitations", []) or []),
            }
        )
        write_json(staging / OBSERVER_ARTIFACTS["manifest"], manifest)
        _publish_observer_tree(staging, output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "AUDIENCE_PROFILES",
    "OBSERVER_AGENT_SCHEMA",
    "OBSERVER_ARTIFACTS",
    "OBSERVER_EPISODE_SCHEMA",
    "OBSERVER_FEED_SCHEMA",
    "OBSERVER_INDEX_SCHEMA",
    "OBSERVER_LINEAGE_SCHEMA",
    "OBSERVER_MANIFEST_SCHEMA",
    "OBSERVER_PREFLIGHT_SCHEMA",
    "OBSERVER_REGION_SCHEMA",
    "OBSERVER_WORLD_SCHEMA",
    "ObserverConfig",
    "ObserverError",
    "load_scientific_source",
    "observer_storage_preflight",
    "render_observer_episode",
    "render_observer_event",
    "select_observer_events",
    "select_observer_episodes",
    "validate_observer_wording",
    "write_observer_outputs",
]
