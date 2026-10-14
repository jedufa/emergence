from __future__ import annotations

import pytest

from simulation.chronicle.observer import (
    ObserverConfig,
    ObserverError,
    _card_summary,
    _cosmic_contextual_humour_candidates,
    render_observer_event,
    validate_observer_wording,
)
from simulation.chronicle.observer_templates import (
    AUDIENCE_PROFILES,
    COSMIC_CARD_HUMOUR_LINES,
    COSMIC_CREATOR_PERSONA,
    COSMIC_HUMOUR_LINES,
    RELIGIOUS_SPECIFICITY_PATTERNS,
    template_inventory,
)


def _event(event_type: str, **measurements):
    return {
        "event_id": f"evt_{event_type}",
        "event_type": event_type,
        "event_category": "ecology" if "food" in event_type or "predator" in event_type else "lineage",
        "generation": 4,
        "step": 10,
        "subject_id": "L2",
        "lineage_ids": ["L2"],
        "agent_ids": [21],
        "measurements": measurements,
        "confidence_class": "measured",
        "causality_class": "not_assessed",
    }


def _entry():
    return {"chronicle_entry_id": "chr_a", "episode_id": "epi_a"}


def test_cosmic_creator_profile_is_optional_and_inventory_exposes_contract() -> None:
    assert "cosmic_creator" in AUDIENCE_PROFILES
    ObserverConfig(audience_profile="cosmic_creator").validate()
    inventory = template_inventory()
    assert "cosmic_creator" in inventory["audience_profiles"]
    assert inventory["cosmic_creator_persona"] == COSMIC_CREATOR_PERSONA
    assert "fictional, non-denominational creator" in COSMIC_CREATOR_PERSONA["role"]
    assert inventory["schema_version"].endswith("v3")


def test_cosmic_creator_event_rendering_is_deterministic_and_source_linked() -> None:
    event = _event("regional_food_depletion", depleted_fraction_before=0.1, depleted_fraction_after=0.8)
    first = render_observer_event(
        event,
        _entry(),
        audience_profile="cosmic_creator",
        episode_family="regional_food_depletion",
        source_completeness="complete",
    )
    second = render_observer_event(
        event,
        _entry(),
        audience_profile="cosmic_creator",
        episode_family="regional_food_depletion",
        source_completeness="complete",
    )
    assert first == second
    assert first["audience_profile"] == "cosmic_creator"
    assert first["presentation_only"] is True
    assert first["source_links"]["event_ids"] == [event["event_id"]]
    assert first["source_links"]["entry_ids"] == ["chr_a"]
    assert first["source_links"]["episode_ids"] == ["epi_a"]
    assert first["confidence_class"] == "measured"
    assert first["causality_class"] == "not_assessed"
    assert first["humour_line"]


def test_cosmic_creator_measurement_reactive_thresholds_are_deterministic() -> None:
    food = _event("regional_food_depletion", depleted_fraction_after=0.78)
    predator = _event("predator_pressure_spike", predator_events=143, predator_damage=0.5)
    bottleneck = _event("population_bottleneck", survival_fraction=0.18)
    near_extinction = _event("lineage_near_extinction", alive_after=1)

    assert _cosmic_contextual_humour_candidates(food, "regional_food_depletion") == _cosmic_contextual_humour_candidates(food, "regional_food_depletion")
    assert any("78.0%" in text for _tid, text in _cosmic_contextual_humour_candidates(food, "regional_food_depletion"))
    assert any("143 predator events" in text for _tid, text in _cosmic_contextual_humour_candidates(predator, "predator_pressure_wave"))
    assert any("18.0%" in text for _tid, text in _cosmic_contextual_humour_candidates(bottleneck, "population_bottleneck"))
    assert any("One living representative" in text for _tid, text in _cosmic_contextual_humour_candidates(near_extinction, "lineage_decline"))


def test_cosmic_templates_have_variation_and_pass_safety_validator() -> None:
    for family, rows in COSMIC_HUMOUR_LINES.items():
        texts = [text for _template_id, text in rows]
        assert len(rows) >= 13, family
        assert len(texts) == len(set(texts)), family
        assert any(".solemn." in template_id for template_id, _text in rows), family
        assert any(".dry." in template_id for template_id, _text in rows), family
        assert any(".sharp." in template_id for template_id, _text in rows), family
        assert any(".audio." in template_id for template_id, _text in rows), family
        assert any(".doc." in template_id for template_id, _text in rows), family
        for _template_id, text in rows:
            validate_observer_wording(text)
    for kind, rows in COSMIC_CARD_HUMOUR_LINES.items():
        assert len(rows) >= 6, kind
        for _template_id, text in rows:
            validate_observer_wording(text)


def test_religious_specificity_and_mental_state_claims_are_rejected() -> None:
    assert "heaven" in RELIGIOUS_SPECIFICITY_PATTERNS
    for forbidden in (
        "The creature planned its future.",
        "The lineage understood the hazard.",
        "A named scripture would not belong here.",
        "No heaven or hell framing belongs here.",
    ):
        with pytest.raises(ObserverError):
            validate_observer_wording(forbidden)


def test_cosmic_mutation_wording_stays_within_gen1_boundary() -> None:
    event = _event("mutation_observation")
    row = render_observer_event(
        event,
        _entry(),
        audience_profile="cosmic_creator",
        source_completeness="complete",
    )
    text = f"{row['plain_factual_text']} {row['humour_line']}".lower()
    assert "beneficial" not in text
    assert "caused success" not in text
    assert "does not establish inheritance" in text
    assert "effect on later outcomes" in text


def test_cosmic_story_cards_and_audio_short_templates_are_available() -> None:
    summary, template_id = _card_summary(
        audience_profile="cosmic_creator",
        kind="lineage",
        identifier="L2",
        event_count=3,
        first_generation=1,
        last_generation=4,
    )
    assert template_id and template_id.startswith("obs3.cosmic.card.lineage")
    assert "Lineage L2 appears" in summary
    validate_observer_wording(summary)
    audio_templates = [
        text
        for rows in COSMIC_HUMOUR_LINES.values()
        for template_id, text in rows
        if ".audio." in template_id
    ]
    assert len(audio_templates) >= 20
    assert max(len(text) for text in audio_templates) <= 90
