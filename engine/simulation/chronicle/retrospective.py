from __future__ import annotations

"""Deterministic post-run Scientific Chronicle regeneration."""

import csv
from pathlib import Path
from typing import Any, Mapping

from simulation.chronicle.config import ChronicleConfig
from simulation.chronicle.episodes import detect_episodes
from simulation.chronicle.recorder import _context_windows
from simulation.chronicle.render import entries_from_events
from simulation.chronicle.schema import MANIFEST_SCHEMA_VERSION, assert_unique_events, canonical_payload, event_order_key, make_event
from simulation.chronicle.storage import (
    ARTIFACT_NAMES,
    artifact_inventory,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def _number(value: Any) -> int | float | str:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        integer = int(text)
        if str(integer) == text or text in {f"+{integer}", f"-{abs(integer)}"}:
            return integer
    except Exception:
        pass
    try:
        result = float(text)
        return result if result == result and abs(result) != float("inf") else 0.0
    except Exception:
        return text


def _metrics_events(metrics_path: Path, *, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    events: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    config = ChronicleConfig(enabled=True, profile="notable")
    for row in rows:
        generation = int(float(row.get("generation", len(events)) or 0))
        selected_keys = (
            "alive",
            "successful_births",
            "deaths_this_gen",
            "failed_births",
            "lineage_effective_alive_lineages",
            "lineage_top1_alive_share",
            "lineage_alive_share_entropy",
            "patch_food_amount_mean",
            "patch_food_depleted_frac",
            "patch_density_hotspot_frac",
            "occupied_patch_waste_mean",
            "jungle_predator_events",
            "jungle_predator_damage_total",
            "jungle_predator_deaths",
            "wdl_delayed_hazard_event_count",
            "wdl_delayed_hazard_damage_total",
            "patch_recovery_positive_total",
        )
        measurements = {key: _number(row[key]) for key in selected_keys if key in row}
        candidates: list[dict[str, Any]] = [
            {
                "event_type": "generation_summary",
                "event_category": "summary",
                "subject_type": "run",
                "subject_id": run_id,
                "phase_order": 900,
                "severity": "informational",
                "live_priority": "background",
                "scientific_importance": 2,
                "historical_importance": 2,
                "measurements": measurements,
                "episode_hint": None,
                "tags": ("retrospective", "generation_summary"),
            }
        ]
        depleted = float(_number(row.get("patch_food_depleted_frac", 0)) or 0)
        previous_depleted = float(_number(previous.get("patch_food_depleted_frac", 0)) or 0)
        if previous_depleted < config.food_depleted_fraction_threshold <= depleted:
            candidates.append(
                {
                    "event_type": "regional_food_depletion",
                    "event_category": "ecology",
                    "subject_type": "world",
                    "subject_id": "global",
                    "phase_order": 500,
                    "severity": "major",
                    "live_priority": "high",
                    "scientific_importance": 4,
                    "historical_importance": 4,
                    "measurements": {"depleted_fraction_before": previous_depleted, "depleted_fraction_after": depleted},
                    "episode_hint": "regional_food_depletion",
                    "tags": ("retrospective", "food"),
                }
            )
        if previous_depleted > config.food_recovery_fraction_threshold >= depleted:
            candidates.append(
                {
                    "event_type": "regional_food_recovery",
                    "event_category": "ecology",
                    "subject_type": "world",
                    "subject_id": "global",
                    "phase_order": 505,
                    "severity": "moderate",
                    "live_priority": "medium",
                    "scientific_importance": 3,
                    "historical_importance": 3,
                    "measurements": {"depleted_fraction_before": previous_depleted, "depleted_fraction_after": depleted},
                    "episode_hint": "ecological_recovery",
                    "tags": ("retrospective", "recovery"),
                }
            )
        predator_events = int(float(row.get("jungle_predator_events", 0) or 0))
        if predator_events > 0:
            candidates.append(
                {
                    "event_type": "predator_pressure_spike",
                    "event_category": "ecology",
                    "subject_type": "population",
                    "subject_id": "global",
                    "phase_order": 530,
                    "severity": "major",
                    "live_priority": "high",
                    "scientific_importance": 4,
                    "historical_importance": 3,
                    "measurements": {
                        "predator_events": predator_events,
                        "predator_damage": float(row.get("jungle_predator_damage_total", 0) or 0),
                    },
                    "episode_hint": "predator_pressure_wave",
                    "tags": ("retrospective", "predator"),
                }
            )
        wdl_events = int(float(row.get("wdl_delayed_hazard_event_count", 0) or 0))
        if wdl_events > 0:
            candidates.append(
                {
                    "event_type": "delayed_hazard_cycle",
                    "event_category": "hazard",
                    "subject_type": "population",
                    "subject_id": "global",
                    "phase_order": 540,
                    "severity": "moderate",
                    "live_priority": "high",
                    "scientific_importance": 4,
                    "historical_importance": 3,
                    "measurements": {"event_count": wdl_events, "damage_total": float(row.get("wdl_delayed_hazard_damage_total", 0) or 0)},
                    "episode_hint": "delayed_hazard_cycle",
                    "tags": ("retrospective", "delayed_hazard"),
                }
            )
        for sequence, candidate in enumerate(sorted(candidates, key=lambda item: int(item["phase_order"]))):
            events.append(
                make_event(
                    run_id=run_id,
                    generation=generation,
                    step=-1,
                    intra_step_sequence=sequence,
                    source_component="retrospective.metrics_csv",
                    source_record_refs=(f"metrics.csv:generation={generation}",),
                    public_interest=0,
                    visual_interest=0,
                    novelty_class="retrospective_derived",
                    confidence_class="derived",
                    causality_class="not_assessed",
                    **candidate,
                )
            )
        previous = row
    return sorted(events, key=event_order_key)


def regenerate_chronicle(run_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    source_root = Path(run_dir).resolve()
    output_root = Path(output_dir).resolve()
    if output_root == source_root or output_root == source_root / "chronicle":
        raise ValueError("retrospective output must be separate from the source run")
    source_event_path = source_root / "chronicle" / ARTIFACT_NAMES["events"]
    source_manifest_path = source_root / "chronicle" / ARTIFACT_NAMES["manifest"]
    metrics_path = source_root / "metrics.csv"
    source_files: list[Path] = []
    limitations: list[str] = []
    if source_event_path.exists():
        events = read_jsonl(source_event_path)
        source_files.append(source_event_path)
        source_completeness = "formal_events_complete"
    elif metrics_path.exists():
        events = _metrics_events(metrics_path, run_id=source_root.name)
        integrity_event = make_event(
            run_id=source_root.name,
            generation=int(events[0].get("generation", 0)) if events else 0,
            step=-1,
            phase_order=990,
            intra_step_sequence=999,
            event_time_scope="run",
            event_type="retrospective_source_incomplete",
            event_category="integrity",
            subject_type="run",
            subject_id=source_root.name,
            severity="major",
            live_priority="high",
            scientific_importance=4,
            historical_importance=4,
            confidence_class="measured",
            causality_class="not_assessed",
            source_component="scientific_chronicle.retrospective",
            source_record_refs=("metrics.csv",),
            measurements={"formal_events_available": False, "metrics_available": True},
            tags=("integrity", "retrospective", "incomplete_source"),
        )
        events.append(integrity_event)
        source_files.append(metrics_path)
        source_completeness = "metrics_only_partial"
        limitations.append("Formal events were unavailable; regeneration is limited to facts present in metrics.csv.")
    else:
        raise FileNotFoundError(f"no formal events or metrics.csv under {source_root}")
    events.sort(key=event_order_key)
    assert_unique_events(events)
    episodes = detect_episodes(events, finalize=True)
    entries = entries_from_events(events, episodes)
    contexts = _context_windows(events, before=8, after=4, max_per_generation=16)
    write_jsonl(output_root / ARTIFACT_NAMES["events"], events)
    write_jsonl(output_root / ARTIFACT_NAMES["entries"], entries)
    write_json(output_root / ARTIFACT_NAMES["episodes"], episodes)
    write_jsonl(output_root / ARTIFACT_NAMES["contexts"], contexts)
    summary = {
        "schema_version": "eiel_scientific_chronicle_summary_v1",
        "run_id": str(events[0].get("run_id", source_root.name)) if events else source_root.name,
        "event_count": len(events),
        "chronicle_entry_count": len(entries),
        "episode_count": len(episodes),
        "source_completeness": source_completeness,
        "limitations": limitations,
        "finalized": True,
    }
    write_json(output_root / ARTIFACT_NAMES["summary"], summary)
    write_json(
        output_root / ARTIFACT_NAMES["index"],
        {
            "schema_version": "eiel_scientific_chronicle_index_v1",
            "run_id": summary["run_id"],
            "generation": max((int(event.get("generation", 0)) for event in events), default=-1),
            "event_count": len(events),
            "episode_count": len(episodes),
            "feeds": {
                "formal_events": ARTIFACT_NAMES["events"],
                "scientific_chronicle": ARTIFACT_NAMES["entries"],
                "episodes": ARTIFACT_NAMES["episodes"],
                "manifest": ARTIFACT_NAMES["manifest"],
            },
            "filter_fields": [
                "event_type",
                "event_category",
                "generation",
                "step",
                "live_priority",
                "subject_type",
                "subject_id",
                "lineage_ids",
                "region_id",
            ],
        },
    )
    sources = [
        {"relative_path": str(path.relative_to(source_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    ]
    source_manifest = dict(read_json(source_manifest_path)) if source_manifest_path.exists() else {}
    inventory = artifact_inventory(
        output_root,
        logical_names=("events", "entries", "episodes", "contexts", "summary", "index"),
    )
    storage = {
        "schema_version": "eiel_scientific_chronicle_storage_summary_v1",
        "profile": "retrospective",
        "artifacts": inventory,
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "event_count": len(events),
        "suppressed_count": 0,
    }
    write_json(output_root / ARTIFACT_NAMES["storage"], storage)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "recorder_version": "scientific_chronicle_retrospective_v1",
        "run_id": summary["run_id"],
        "source_commit": str(source_manifest.get("source_commit") or "unknown"),
        "parent_checkpoint_id": source_manifest.get("parent_checkpoint_id"),
        "profile": "retrospective",
        "event_count": len(events),
        "episode_count": len(episodes),
        "storage_bytes": int(storage["total_bytes"]),
        "source_completeness": source_completeness,
        "source_files": sources,
        "deterministic_generation_mode": "retrospective_canonical_v1",
        "limitations": limitations,
        "finalized": True,
    }
    write_json(output_root / ARTIFACT_NAMES["manifest"], manifest)
    return canonical_payload({"output_dir": str(output_root), **summary, "source_files": sources})
