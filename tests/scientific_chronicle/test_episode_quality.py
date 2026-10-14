from __future__ import annotations

import csv
import json
from pathlib import Path

from simulation.chronicle.compare import _first_meaningful_divergence
from simulation.chronicle.episodes import detect_episodes
from simulation.chronicle.quality import (
    EPISODE_CONTRACTS,
    _interval_openings,
    _opening_metric_valid,
    _source_supports_episode_judgment,
    analyze_validation_matrix,
)
from simulation.chronicle.recorder import _context_windows
from simulation.chronicle.schema import make_event
from simulation.chronicle.storage import ARTIFACT_NAMES, write_json, write_jsonl


def _event(*, generation: int, event_type: str, category: str, region: str | None = None, **measurements):
    return make_event(
        run_id="quality-run",
        generation=generation,
        step=9,
        phase_order=500 if event_type != "generation_summary" else 900,
        intra_step_sequence=0,
        event_type=event_type,
        event_category=category,
        subject_type="world" if category == "ecology" else "run",
        subject_id=region or "global",
        region_id=region,
        severity="major" if category == "ecology" else "informational",
        live_priority="high" if category == "ecology" else "background",
        scientific_importance=4 if category == "ecology" else 2,
        historical_importance=3 if category == "ecology" else 2,
        source_component="test",
        source_record_refs=(f"metrics.csv:generation={generation}",),
        measurements=measurements,
    )


def _write_run(root: Path) -> None:
    chronicle = root / "quality-run" / "chronicle"
    events = [
        _event(
            generation=0,
            event_type="predator_pressure_spike",
            category="ecology",
            predator_events=8,
            predator_damage=0.8,
            predator_deaths=0,
            threshold=1,
        ),
        _event(generation=0, event_type="generation_summary", category="summary", alive=10, predator_events=8),
        _event(generation=1, event_type="generation_summary", category="summary", alive=10, predator_events=7),
        _event(generation=2, event_type="generation_summary", category="summary", alive=10, predator_events=0),
    ]
    events.sort(key=lambda item: (item["generation"], item["step"], item["phase_order"], item["event_id"]))
    episodes = detect_episodes(events, finalize=True)
    write_jsonl(chronicle / ARTIFACT_NAMES["events"], events)
    write_jsonl(chronicle / ARTIFACT_NAMES["entries"], [])
    write_json(chronicle / ARTIFACT_NAMES["episodes"], episodes)
    write_jsonl(chronicle / ARTIFACT_NAMES["contexts"], [])
    write_json(chronicle / ARTIFACT_NAMES["summary"], {})
    write_json(
        chronicle / ARTIFACT_NAMES["manifest"],
        {
            "source_completeness": "live_generation_boundary_complete",
            "config": {"episode_gap_generations": 1},
        },
    )
    write_json(chronicle / ARTIFACT_NAMES["storage"], {})
    write_json(chronicle / ARTIFACT_NAMES["index"], {})
    metrics_path = root / "quality-run" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("generation", "alive", "jungle_predator_events", "cohort_survival_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            (
                {"generation": 0, "alive": 10, "jungle_predator_events": 8, "cohort_survival_rate": 1.0},
                {"generation": 1, "alive": 10, "jungle_predator_events": 7, "cohort_survival_rate": 1.0},
                {"generation": 2, "alive": 10, "jungle_predator_events": 0, "cohort_survival_rate": 1.0},
            )
        )


def test_episode_semantic_contracts_cover_all_v1_families() -> None:
    assert set(EPISODE_CONTRACTS) == {
        "regional_food_depletion",
        "crowding_waste_crisis",
        "predator_pressure_wave",
        "ecological_recovery",
        "lineage_expansion",
        "lineage_decline",
        "lineage_extinction_sequence",
        "population_bottleneck",
        "delayed_hazard_cycle",
    }
    for contract in EPISODE_CONTRACTS.values():
        assert contract["opening_rule"]
        assert contract["continuation_rule"]
        assert contract["closing_rule"]
        assert contract["source_completeness_requirement"]


def test_aggregate_only_source_never_supports_lineage_episode_judgment() -> None:
    completeness = "live_generation_boundary_native_aggregate_lifecycle"

    assert _source_supports_episode_judgment("lineage_decline", completeness) is False
    assert _source_supports_episode_judgment("lineage_expansion", completeness) is False
    assert _source_supports_episode_judgment("predator_pressure_wave", completeness) is True


def test_food_opening_review_respects_depletion_recovery_hysteresis() -> None:
    metrics = {
        0: {"patch_food_depleted_frac": 0.30},
        1: {"patch_food_depleted_frac": 0.20},
        2: {"patch_food_depleted_frac": 0.28},
        3: {"patch_food_depleted_frac": 0.05},
        4: {"patch_food_depleted_frac": 0.27},
    }
    config = {"food_depleted_fraction_threshold": 0.25, "food_recovery_fraction_threshold": 0.10}

    openings = _interval_openings("regional_food_depletion", metrics, config)

    assert openings == [0, 4]


def test_opening_review_is_specific_to_the_formal_trigger() -> None:
    metrics = {
        0: {
            "patch_food_depleted_frac": 0.15,
            "patch_density_hotspot_frac": 0.40,
            "occupied_patch_waste_mean": 0.0,
        }
    }
    config = {
        "food_depleted_fraction_threshold": 0.25,
        "crowding_hotspot_fraction_threshold": 0.20,
        "waste_hotspot_threshold": 0.25,
    }
    food = _event(
        generation=0,
        event_type="regional_food_depletion",
        category="ecology",
        depleted_fraction_after=0.15,
    )
    waste = _event(
        generation=0,
        event_type="waste_hotspot",
        category="ecology",
        occupied_waste_after=0.0,
    )

    assert _opening_metric_valid("regional_food_depletion", food, metrics, config) is False
    assert _opening_metric_valid("crowding_waste_crisis", waste, metrics, config) is False


def test_quality_analysis_accepts_summary_grounded_pressure_boundary(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "cells": [
                    {"run_id": "quality-run", "scenario": "predator", "seed": 1}
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"

    result = analyze_validation_matrix(matrix, runs, output)

    assert result["classification_counts"]["correct"] == 1
    assert result["live_retrospective_accepted"] is True
    quality = json.loads((output / "episode_quality_summary.json").read_text(encoding="utf-8"))
    predator = quality["families"]["predator_pressure_wave"]
    assert predator["seed_recurrence_count"] == 1
    assert predator["seeds"] == [1]
    negative = quality["families"]["regional_food_depletion"]["representative_examples"][
        "negative_or_control"
    ]
    assert negative["absence_confirmed"] is True
    assert negative["independent_metric_opening_count"] == 0
    for path in output.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_episode_detector_keeps_regions_separate() -> None:
    events = [
        _event(generation=0, event_type="waste_hotspot", category="ecology", region="region:a", threshold=0.25),
        _event(generation=0, event_type="waste_hotspot", category="ecology", region="region:b", threshold=0.25),
    ]

    episodes = detect_episodes(events, finalize=True)

    assert len(episodes) == 2
    assert {episode["affected_region"] for episode in episodes} == {"region:a", "region:b"}


def test_interval_episode_identity_is_stable_through_finalization() -> None:
    events = [
        _event(
            generation=0,
            event_type="predator_pressure_spike",
            category="ecology",
            predator_events=8,
            threshold=1,
        ),
        _event(generation=0, event_type="generation_summary", category="summary", predator_events=8),
        _event(generation=1, event_type="generation_summary", category="summary", predator_events=7),
        _event(generation=2, event_type="generation_summary", category="summary", predator_events=0),
    ]
    events.sort(key=lambda item: (item["generation"], item["step"], item["phase_order"], item["event_id"]))

    provisional = detect_episodes(events[:3], finalize=False)
    finalized = detect_episodes(events, finalize=True)

    assert provisional[0]["status"] == "open"
    assert provisional[0]["episode_id"] == finalized[0]["episode_id"]
    assert provisional[0]["end_generation"] == 1
    assert set(provisional[0]["source_event_ids"]).issubset(finalized[0]["source_event_ids"])
    assert finalized[0]["status"] == "closed"
    assert finalized[0]["outcome"] == "condition_cleared"
    assert finalized[0]["end_generation"] == 2


def test_future_component_trigger_does_not_extend_prior_crisis() -> None:
    events = [
        _event(
            generation=0,
            event_type="crowding_hotspot",
            category="ecology",
            threshold=0.20,
        ),
        _event(
            generation=0,
            event_type="generation_summary",
            category="summary",
            crowding_hotspot_fraction=0.30,
            occupied_waste_mean=0.0,
        ),
        _event(
            generation=1,
            event_type="generation_summary",
            category="summary",
            crowding_hotspot_fraction=0.0,
            occupied_waste_mean=0.30,
        ),
        _event(
            generation=2,
            event_type="waste_hotspot",
            category="ecology",
            threshold=0.25,
        ),
        _event(
            generation=2,
            event_type="generation_summary",
            category="summary",
            crowding_hotspot_fraction=0.0,
            occupied_waste_mean=0.30,
        ),
        _event(
            generation=3,
            event_type="generation_summary",
            category="summary",
            crowding_hotspot_fraction=0.0,
            occupied_waste_mean=0.0,
        ),
    ]
    events.sort(key=lambda item: (item["generation"], item["step"], item["phase_order"], item["event_id"]))

    episodes = detect_episodes(events, finalize=True)

    assert [(episode["start_generation"], episode["end_generation"]) for episode in episodes] == [
        (0, 1),
        (2, 3),
    ]


def test_predator_continuation_recomputes_population_scaled_threshold() -> None:
    events = [
        _event(
            generation=0,
            event_type="predator_pressure_spike",
            category="ecology",
            predator_events=5,
            threshold=5,
        ),
        _event(
            generation=0,
            event_type="generation_summary",
            category="summary",
            alive=100,
            predator_events=5,
        ),
        _event(
            generation=1,
            event_type="generation_summary",
            category="summary",
            alive=10,
            predator_events=2,
        ),
        _event(
            generation=2,
            event_type="generation_summary",
            category="summary",
            alive=10,
            predator_events=0,
        ),
    ]
    events.sort(key=lambda item: (item["generation"], item["step"], item["phase_order"], item["event_id"]))

    episode = detect_episodes(events, finalize=True)[0]

    assert episode["start_generation"] == 0
    assert episode["end_generation"] == 2


def test_transition_only_food_recovery_closes_at_trigger_generation() -> None:
    events = [
        _event(
            generation=1,
            event_type="regional_food_recovery",
            category="ecology",
            depleted_fraction_after=0.05,
            threshold=0.10,
        ),
        _event(
            generation=1,
            event_type="generation_summary",
            category="summary",
            recovery_positive_total=0.0,
        ),
    ]
    events.sort(key=lambda item: (item["generation"], item["step"], item["phase_order"], item["event_id"]))

    episode = detect_episodes(events, finalize=False)[0]

    assert episode["status"] == "closed"
    assert episode["start_generation"] == episode["end_generation"] == 1
    assert episode["outcome"] == "recovery_transition_recorded"


def test_first_divergence_ignores_routine_event_before_meaningful_event() -> None:
    events_a = [
        _event(generation=0, event_type="notable_birth", category="lifecycle"),
        _event(generation=0, event_type="generation_summary", category="summary", alive=20),
        _event(generation=1, event_type="predator_pressure_spike", category="ecology", predator_events=4),
        _event(generation=1, event_type="generation_summary", category="summary", alive=20),
    ]
    events_b = [
        _event(generation=0, event_type="generation_summary", category="summary", alive=20),
        _event(generation=1, event_type="generation_summary", category="summary", alive=20),
    ]

    divergence = _first_meaningful_divergence(events_a, events_b)

    assert divergence["status"] == "diverged"
    assert divergence["generation"] == 1
    assert divergence["event_a"]["event_type"] == "predator_pressure_spike"
    assert "meaningful structured event" in divergence["reason"]


def test_first_divergence_rejects_minor_summary_noise() -> None:
    events_a = [
        _event(
            generation=0,
            event_type="generation_summary",
            category="summary",
            alive=20,
            patch_food_depleted_frac=0.20,
        )
    ]
    events_b = [
        _event(
            generation=0,
            event_type="generation_summary",
            category="summary",
            alive=19,
            patch_food_depleted_frac=0.25,
        )
    ]

    divergence = _first_meaningful_divergence(events_a, events_b)

    assert divergence["status"] == "no_meaningful_divergence"
    assert divergence["generation"] is None


def test_first_divergence_accepts_material_summary_difference() -> None:
    events_a = [_event(generation=0, event_type="generation_summary", category="summary", alive=20)]
    events_b = [_event(generation=0, event_type="generation_summary", category="summary", alive=17)]

    divergence = _first_meaningful_divergence(events_a, events_b)

    assert divergence["status"] == "diverged"
    assert divergence["generation"] == 0
    assert divergence["metric_differences"][0]["metric"] == "alive"


def test_first_divergence_includes_user_visible_event_classification() -> None:
    event_a = _event(
        generation=0,
        event_type="predator_pressure_spike",
        category="ecology",
        predator_events=4,
    )
    event_b = dict(event_a)
    event_b["severity"] = "critical"
    event_b["live_priority"] = "critical"

    divergence = _first_meaningful_divergence([event_a], [event_b])

    assert divergence["status"] == "diverged"
    assert divergence["generation"] == 0
    assert divergence["event_a"]["severity"] == "major"
    assert divergence["event_b"]["severity"] == "critical"


def test_overlapping_context_windows_remain_bounded_and_deduplicated() -> None:
    events = [
        make_event(
            run_id="context-run",
            generation=0,
            step=index,
            phase_order=500,
            intra_step_sequence=index,
            event_type="pressure_observation",
            event_category="ecology",
            subject_type="patch",
            subject_id=str(index),
            severity="major",
            live_priority="high",
            scientific_importance=4,
            historical_importance=3,
            source_component="test",
            source_record_refs=(f"test:{index}",),
        )
        for index in range(40)
    ]

    windows = _context_windows(events, before=2, after=2, max_per_generation=3)

    assert 0 < len(windows) <= 3
    assert len({window["context_window_id"] for window in windows}) == len(windows)
    assert all(len(window["source_event_ids"]) == len(set(window["source_event_ids"])) for window in windows)
