from __future__ import annotations

import copy

import pytest

from native_runtime.patch_public_reporting import _validated_wdl_public_metrics
from native_runtime.public_metrics_overlay import public_metrics_overlay
from tests.perf.test_eiel_native_patch_public_reporting import (
    _patch_raw_result,
    _patch_reporting_inputs,
)


def _raw_wdl_metrics() -> dict[str, object]:
    return {
        "wdl_delayed_hazard_enable": 1,
        "wdl_delayed_hazard_cue_cell_count": 1.5,
        "wdl_delayed_hazard_active_cell_count": 2.0,
        "wdl_delayed_hazard_event_count": 2,
        "wdl_delayed_hazard_agent_cue_exposure_count": 4,
        "wdl_delayed_hazard_agent_active_exposure_count": 3,
        "wdl_delayed_hazard_damage_total": 0.75,
        "wdl_delayed_hazard_damage_mean": 0.25,
        "wdl_delayed_hazard_damage_max": 0.25,
        "wdl_delayed_hazard_agents_damaged": 2,
        "wdl_delayed_hazard_cue_to_active_delay_mean": 2.0,
        "wdl_delayed_hazard_cue_without_active_count": 1,
        "wdl_delayed_hazard_active_without_prior_cue_count": 0,
        "wdl_cue_seen_agents": 4,
        "wdl_cue_exposed_next_move_away_count": 2,
        "wdl_cue_exposed_next_move_toward_count": 1,
        "wdl_cue_exposed_stayed_count": 1,
        "wdl_pre_hazard_exit_count": 2,
        "wdl_hazard_enter_count": 1,
        "wdl_hazard_avoidance_proxy": 0.5,
        "wdl_hazard_exposure_after_cue_count": 2,
        "wdl_hazard_damage_by_lineage_json": '{"10":0.5,"20":0.25}',
        "wdl_cue_exposure_by_lineage_json": "{}",
        "wdl_hazard_exposure_by_lineage_json": '{"10":1.0,"20":1.0}',
    }


def test_native_wdl_public_reporting_accepts_the_exact_24_column_contract() -> None:
    metrics = _validated_wdl_public_metrics(_raw_wdl_metrics())

    assert len(metrics) == 24
    assert metrics["wdl_delayed_hazard_enable"] == 1
    assert metrics["wdl_delayed_hazard_event_count"] == 2.0
    assert metrics["wdl_delayed_hazard_damage_mean"] == 0.25
    assert metrics["wdl_hazard_damage_by_lineage_json"] == '{"10":0.5,"20":0.25}'


def test_active_wdl_native_result_reaches_the_production_public_overlay() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P1")
    env_cfg.wdl_delayed_hazard_enable = True
    env_cfg.wdl_delayed_hazard_observation_enable = False
    projection["metadata"]["capability_mask"]["wdl"] = True
    projection["metadata"]["native_runtime_wdl_execution_enable"] = True
    raw = _patch_raw_result(behavior_rows)
    raw.update(_raw_wdl_metrics())

    overlay = public_metrics_overlay(
        raw,
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    expected = _validated_wdl_public_metrics(raw)
    assert {name: overlay[name] for name in expected} == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("wdl_delayed_hazard_enable", 0, "requires wdl_delayed_hazard_enable=1"),
        (
            "wdl_delayed_hazard_damage_mean",
            0.5,
            "damage mean contradicts total/exposure count",
        ),
        (
            "wdl_hazard_avoidance_proxy",
            0.25,
            "avoidance proxy contradicts action counts",
        ),
        (
            "wdl_hazard_damage_by_lineage_json",
            '{"10":0.25}',
            "lineage damage contradicts total damage",
        ),
        (
            "wdl_hazard_exposure_by_lineage_json",
            '{"20":1.0,"10":1.0}',
            "is not canonical JSON",
        ),
    ),
)
def test_native_wdl_public_reporting_fails_closed_on_contradictions(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = copy.deepcopy(_raw_wdl_metrics())
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        _validated_wdl_public_metrics(raw)


def test_native_wdl_public_reporting_rejects_missing_fields() -> None:
    raw = _raw_wdl_metrics()
    del raw["wdl_hazard_enter_count"]

    with pytest.raises(ValueError, match="wdl_hazard_enter_count"):
        _validated_wdl_public_metrics(raw)
