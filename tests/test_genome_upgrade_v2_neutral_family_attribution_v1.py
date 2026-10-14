from __future__ import annotations

import pytest

from engine.evolution.genome_upgrade_v2_neutral_family_attribution_v1 import (
    MARKER_ID,
    NeutralFamilyAttributionError,
    NeutralFamilyEventV1,
    NeutralFamilyWindowV1,
    attribute_neutral_families_v1,
    build_schedule_payload_v1,
)


def test_neutral_family_introduces_and_reaches_descendant() -> None:
    windows = (
        NeutralFamilyWindowV1(
            family_id="early",
            opens_at_tick=0,
            closes_at_tick=160,
            eligible_event_kind="ordinary_birth",
        ),
    )
    receipt = attribute_neutral_families_v1(
        cell_id="stage1s-seed9401",
        paired_seed=9401,
        windows=windows,
        events=(
            NeutralFamilyEventV1(
                cell_id="stage1s-seed9401",
                generation=0,
                tick=8,
                event_kind="ordinary_birth",
                parent_agent_ids=(10, 11),
                child_agent_id=64,
                slot=64,
            ),
            NeutralFamilyEventV1(
                cell_id="stage1s-seed9401",
                generation=1,
                tick=4,
                event_kind="ordinary_birth",
                parent_agent_ids=(64, 12),
                child_agent_id=65,
                slot=65,
            ),
        ),
    )

    assert receipt["marker_id"] == MARKER_ID
    assert receipt["introduced_family_count"] == 1
    assert receipt["introduction_records"][0]["child_agent_id"] == 64
    descendant = receipt["transmission_records"][1]
    assert descendant["family_id"] == "early"
    assert descendant["inherited"] is True
    assert descendant["introduced"] is False
    assert receipt["terminal_carriers_by_agent_id"] == {
        "64": ["early"],
        "65": ["early"],
    }
    assert receipt["neutrality_contract"]["policy_input"] is False


def test_neutral_family_extinction_and_biological_id_reset() -> None:
    windows = (
        NeutralFamilyWindowV1(
            family_id="late",
            opens_at_tick=0,
            closes_at_tick=80,
            eligible_event_kind="turnover_refill_birth",
        ),
    )
    receipt = attribute_neutral_families_v1(
        cell_id="legacy-seed9401",
        paired_seed=9401,
        windows=windows,
        events=(
            NeutralFamilyEventV1(
                cell_id="legacy-seed9401",
                generation=0,
                tick=79,
                event_kind="turnover_refill_birth",
                parent_agent_ids=(20, 21),
                child_agent_id=64,
                slot=3,
            ),
            NeutralFamilyEventV1(
                cell_id="legacy-seed9401",
                generation=1,
                tick=0,
                event_kind="death",
                child_agent_id=64,
                slot=3,
                reason="turnover_cull",
            ),
            NeutralFamilyEventV1(
                cell_id="legacy-seed9401",
                generation=1,
                tick=1,
                event_kind="turnover_refill_birth",
                parent_agent_ids=(22, 23),
                child_agent_id=65,
                slot=3,
            ),
        ),
    )

    assert receipt["terminal_carriers_by_agent_id"] == {}
    assert receipt["extinction_records"] == [
        {
            "cell_id": "legacy-seed9401",
            "generation": 1,
            "tick": 0,
            "physical_tick": 80,
            "event_kind": "death",
            "child_agent_id": 64,
            "parent_agent_ids": [],
            "slot": 3,
            "reason": "turnover_cull",
            "family_id": "late",
            "extinction_reason": "turnover_cull",
        }
    ]


def test_missing_introduction_fails_closed() -> None:
    windows = (
        NeutralFamilyWindowV1(
            family_id="middle",
            opens_at_tick=80,
            closes_at_tick=120,
            eligible_event_kind="ordinary_birth",
        ),
    )
    with pytest.raises(NeutralFamilyAttributionError, match="middle"):
        attribute_neutral_families_v1(
            cell_id="stage1s-seed9402",
            paired_seed=9402,
            windows=windows,
            events=(
                NeutralFamilyEventV1(
                    cell_id="stage1s-seed9402",
                    generation=0,
                    tick=10,
                    event_kind="ordinary_birth",
                    child_agent_id=64,
                    parent_agent_ids=(1, 2),
                ),
            ),
        )


def test_multicell_leakage_is_rejected() -> None:
    windows = (
        NeutralFamilyWindowV1(
            family_id="early",
            opens_at_tick=0,
            closes_at_tick=80,
            eligible_event_kind="ordinary_birth",
        ),
    )
    with pytest.raises(NeutralFamilyAttributionError, match="cell_id"):
        attribute_neutral_families_v1(
            cell_id="cell-a",
            paired_seed=1,
            windows=windows,
            events=(
                NeutralFamilyEventV1(
                    cell_id="cell-b",
                    generation=0,
                    tick=1,
                    event_kind="ordinary_birth",
                    child_agent_id=64,
                    parent_agent_ids=(1, 2),
                ),
            ),
        )


def test_schedule_payload_is_stable_and_explicit() -> None:
    window = NeutralFamilyWindowV1(
        family_id="early",
        opens_at_tick=0,
        closes_at_tick=80,
        eligible_event_kind="ordinary_birth",
    )
    first = build_schedule_payload_v1(
        cell_id="cell-a",
        paired_seed=1,
        windows=(window,),
    )
    second = build_schedule_payload_v1(
        cell_id="cell-a",
        paired_seed=1,
        windows=(window,),
    )

    assert first == second
    assert first["schema_version"] == "genome_upgrade_v2_neutral_family_schedule_v1"
    assert first["marker_id"] == MARKER_ID
    assert first["schedule_sha256"]
