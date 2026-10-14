from __future__ import annotations

import math

import pytest

from simulation.chronicle.episodes import detect_episodes
from simulation.chronicle.query import filter_events
from simulation.chronicle.render import entry_from_event, render_scientific_summary
from simulation.chronicle.schema import ChronicleSchemaError, canonical_json, make_event, validate_event


def _event(*, event_type: str, generation: int, sequence: int, subject_id: str = "global") -> dict:
    return make_event(
        run_id="run-a",
        branch_id="branch-a",
        generation=generation,
        step=9,
        phase_order=500,
        intra_step_sequence=sequence,
        event_type=event_type,
        event_category="ecology" if not event_type.startswith("lineage_") else "lineage",
        subject_type="lineage" if event_type.startswith("lineage_") else "world",
        subject_id=subject_id,
        lineage_ids=(subject_id,) if event_type.startswith("lineage_") else (),
        severity="major",
        live_priority="high",
        scientific_importance=4,
        historical_importance=4,
        source_component="test",
        source_record_refs=(f"test:g{generation}",),
        measurements={"value": generation + 1.0},
    )


def test_event_id_and_canonical_serialization_are_deterministic() -> None:
    first = _event(event_type="regional_food_depletion", generation=2, sequence=0)
    second = _event(event_type="regional_food_depletion", generation=2, sequence=0)

    assert first["event_id"] == second["event_id"]
    assert canonical_json(first) == canonical_json(second)
    validate_event(first)


def test_event_id_changes_with_stable_identity_fields() -> None:
    first = _event(event_type="regional_food_depletion", generation=2, sequence=0)
    second = _event(event_type="regional_food_depletion", generation=3, sequence=0)

    assert first["event_id"] != second["event_id"]


def test_non_finite_measurements_are_rejected() -> None:
    with pytest.raises(ChronicleSchemaError, match="non-finite"):
        make_event(
            run_id="r",
            generation=0,
            step=0,
            phase_order=1,
            intra_step_sequence=0,
            event_type="generation_summary",
            event_category="summary",
            subject_type="run",
            source_component="test",
            measurements={"bad": math.nan},
        )


def test_event_category_is_a_bounded_user_visible_enum() -> None:
    event = _event(event_type="regional_food_depletion", generation=2, sequence=0)
    event["event_category"] = "drama"

    with pytest.raises(ChronicleSchemaError, match="invalid event_category"):
        validate_event(event)


def test_episode_detection_is_deterministic_and_conservative() -> None:
    events = [
        _event(event_type="regional_food_depletion", generation=1, sequence=0),
        _event(event_type="regional_food_depletion", generation=2, sequence=0),
        _event(event_type="regional_food_depletion", generation=6, sequence=0),
    ]

    live = detect_episodes(events, gap_generations=1, finalize=False)
    final = detect_episodes(events, gap_generations=1, finalize=True)

    assert [episode["status"] for episode in live] == ["closed", "open"]
    assert [episode["status"] for episode in final] == ["closed", "closed"]
    assert final[0]["source_event_ids"] == [events[0]["event_id"], events[1]["event_id"]]
    assert final == detect_episodes(events, gap_generations=1, finalize=True)
    assert all(episode["causality_class"] == "temporal_association" for episode in final)


def test_lineage_extinction_closes_its_episode_immediately() -> None:
    event = _event(event_type="lineage_extinction", generation=4, sequence=0, subject_id="L7")
    episodes = detect_episodes([event], finalize=False)

    assert len(episodes) == 1
    assert episodes[0]["episode_type"] == "lineage_extinction_sequence"
    assert episodes[0]["status"] == "closed"
    assert episodes[0]["outcome"] == "lineage_extinction_recorded"


def test_query_filters_structured_fields_in_total_order() -> None:
    events = [
        _event(event_type="lineage_extinction", generation=3, sequence=0, subject_id="L7"),
        _event(event_type="regional_food_depletion", generation=1, sequence=0),
        _event(event_type="lineage_rapid_share_decline", generation=2, sequence=0, subject_id="L8"),
    ]

    selected = filter_events(
        events,
        categories=("lineage",),
        lineage_ids=("L7",),
        generation_start=2,
        minimum_scientific_importance=4,
    )

    assert [event["event_type"] for event in selected] == ["lineage_extinction"]


def test_scientific_rendering_does_not_mutate_formal_event() -> None:
    event = _event(event_type="regional_food_depletion", generation=2, sequence=0)
    before = canonical_json(event)

    entry = entry_from_event(event)

    assert canonical_json(event) == before
    assert entry["source_event_ids"] == [event["event_id"]]


def test_scientific_sentence_is_event_specific_and_avoids_mental_state_claims() -> None:
    event = make_event(
        run_id="r",
        generation=4,
        step=9,
        phase_order=410,
        intra_step_sequence=0,
        event_type="lineage_dominance_crossing",
        event_category="lineage",
        subject_type="lineage",
        subject_id="L7",
        lineage_ids=("L7",),
        severity="major",
        live_priority="high",
        scientific_importance=4,
        historical_importance=4,
        source_component="test",
        measurements={"share_before": 0.42, "share_after": 0.53, "threshold": 0.5},
    )

    sentence = render_scientific_summary(event)

    assert sentence == (
        "Lineage L7 increased from share 0.42 to 0.53, crossing the configured dominance "
        "threshold 0.5."
    )
    for prohibited in ("thought", "intended", "planned", "remembered", "understood", "conscious"):
        assert prohibited not in sentence.lower()
