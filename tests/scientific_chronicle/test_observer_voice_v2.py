from __future__ import annotations

from simulation.chronicle.observer import (
    _contextual_humour_candidates,
    _event_plain,
    validate_observer_wording,
)
from simulation.chronicle.observer_templates import (
    CARD_HUMOUR_LINES,
    DRY_HUMOUR_LINES,
    EVENT_HEADLINES,
    NARRATOR_PERSONA,
    PROFILE_FRAMES,
    template_inventory,
)


def _event(event_type: str, **measurements):
    return {
        "event_type": event_type,
        "event_category": "ecology" if "food" in event_type else "lineage",
        "generation": 4,
        "step": 10,
        "subject_id": "L2",
        "lineage_ids": ["L2"],
        "agent_ids": [21],
        "measurements": measurements,
    }


def test_persona_and_contract_are_exposed() -> None:
    inventory = template_inventory()
    assert inventory["narrator_persona"] == NARRATOR_PERSONA
    assert "overqualified field-operations narrator" in NARRATOR_PERSONA["role"]
    assert inventory["schema_version"].endswith("v3")


def test_each_dry_family_has_real_variation() -> None:
    for family, rows in DRY_HUMOUR_LINES.items():
        texts = [text for _template_id, text in rows]
        assert len(rows) >= 6, family
        assert len(texts) == len(set(texts)), family
    for kind, rows in CARD_HUMOUR_LINES.items():
        texts = [text for _template_id, text in rows]
        assert len(rows) >= 6, kind
        assert len(texts) == len(set(texts)), kind


def test_extinction_wording_adds_angle_instead_of_repeating_zero() -> None:
    plain = _event_plain(_event("lineage_extinction"))
    assert plain == "Lineage L2 has no living representatives."
    lines = [text for _template_id, text in DRY_HUMOUR_LINES["lineage_extinction_sequence"]]
    assert all("little ambiguity" not in text.lower() for text in lines)
    assert all("no living representatives" not in text.lower() for text in lines)


def test_public_plain_is_direct_not_bureaucratic() -> None:
    birth = _event_plain(_event("notable_birth"))
    death = _event_plain(_event("notable_death"))
    assert "was born" in birth
    assert "died" in death
    assert "recorded as born" not in birth
    assert "recorded as dead" not in death


def test_measurement_reactive_candidates_are_added_deterministically() -> None:
    event = _event(
        "regional_food_depletion",
        depleted_fraction_before=0.1,
        depleted_fraction_after=0.8,
    )
    first = _contextual_humour_candidates(event, "regional_food_depletion")
    second = _contextual_humour_candidates(event, "regional_food_depletion")
    assert first == second
    assert any("historical document" in text for _template_id, text in first)


def test_all_generated_template_text_respects_claims_contract() -> None:
    for rows in DRY_HUMOUR_LINES.values():
        for _template_id, text in rows:
            validate_observer_wording(text)
    for rows in CARD_HUMOUR_LINES.values():
        for _template_id, text in rows:
            validate_observer_wording(text)
    for headline in EVENT_HEADLINES.values():
        validate_observer_wording(headline.format(generation=1, subject="L1"))
    for rows in PROFILE_FRAMES.values():
        for _template_id, frame in rows:
            validate_observer_wording(frame.format(generation=1, plain="A fact.", humour="A line."))
