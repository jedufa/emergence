from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from simulation.metrics_pkg.path1_drh0 import (
    PATH1_DRH0_METRIC_FIELDS,
    path1_drh0_generation_metrics,
)
from simulation.metrics_pkg.rows import MetricsRow


def _cfg(*, reporting: bool = True, reachability: bool = True, observation: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.patch_world_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.path1_drh0_reporting_enable = bool(reporting)
    cfg.path1_drh0_refuge_reachability_enable = bool(reachability)
    cfg.path1_drh0_observation_enable = bool(observation)
    cfg.path1_drh0_refuge_radius = 2
    cfg.path1_drh0_min_refuge_cells = 1
    return cfg


def _env(cfg: EnvConfig, *, cue: tuple[int, int] | None = (2, 2), refuge: tuple[int, int] | None = (2, 3)):
    shape = (5, 5)
    cue_map = np.zeros(shape, dtype=np.float32)
    timer = np.full(shape, -1, dtype=np.int32)
    if cue is not None:
        cue_map[cue] = 1.0
        timer[cue] = 2
    cover = np.zeros(shape, dtype=np.float32)
    shelter = np.zeros(shape, dtype=np.float32)
    if refuge is not None:
        cover[refuge] = 1.0
        shelter[refuge] = 0.5
    agent_alive = np.zeros((4,), dtype=bool)
    agent_pos_y = np.full((4,), -1, dtype=np.int32)
    agent_pos_x = np.full((4,), -1, dtype=np.int32)
    if cue is not None:
        agent_alive[0] = True
        agent_pos_y[0], agent_pos_x[0] = cue
    if refuge is not None:
        agent_alive[1] = True
        agent_pos_y[1], agent_pos_x[1] = refuge
    return SimpleNamespace(
        cfg=cfg,
        patch_delayed_hazard_cue=cue_map,
        patch_delayed_hazard_active=np.zeros(shape, dtype=np.float32),
        patch_delayed_hazard_prior_cue=np.zeros(shape, dtype=np.int8),
        patch_delayed_hazard_timer=timer,
        patch_cover=cover,
        patch_shelter=shelter,
        agent_alive=agent_alive,
        agent_pos_y=agent_pos_y,
        agent_pos_x=agent_pos_x,
    )


def test_path1_drh0_metrics_are_in_metrics_row_schema() -> None:
    row_fields = {field.name for field in fields(MetricsRow)}
    assert set(PATH1_DRH0_METRIC_FIELDS).issubset(row_fields)


def test_path1_drh0_default_off_emits_no_nonzero_metrics() -> None:
    metrics = path1_drh0_generation_metrics(_env(_cfg(reporting=False)))
    assert set(metrics) == set(PATH1_DRH0_METRIC_FIELDS)
    assert all(value == pytest.approx(0.0) for value in metrics.values())


def test_path1_drh0_no_cue_control_stays_zero() -> None:
    metrics = path1_drh0_generation_metrics(_env(_cfg(), cue=None, refuge=(2, 3)))
    assert metrics["path1_refuge_cue_cell_count"] == pytest.approx(0.0)
    assert metrics["path1_refuge_zone_cell_count"] == pytest.approx(0.0)
    assert metrics["path1_refuge_agents_reachable_refuge_at_cue"] == pytest.approx(0.0)


def test_path1_drh0_no_active_control_reports_opportunity_without_consequence() -> None:
    metrics = path1_drh0_generation_metrics(_env(_cfg()))
    assert metrics["path1_refuge_cue_cell_count"] == pytest.approx(1.0)
    assert metrics["path1_refuge_active_cell_count"] == pytest.approx(0.0)
    assert metrics["path1_refuge_zone_cell_count"] == pytest.approx(1.0)
    assert metrics["path1_refuge_agents_reachable_refuge_at_cue"] == pytest.approx(1.0)
    assert metrics["path1_refuge_agents_in_refuge_before_active"] == pytest.approx(1.0)
    assert metrics["path1_drh0_refuge_cell_count"] == pytest.approx(1.0)
    assert metrics["path1_drh0_reachable_refuge_count"] == pytest.approx(1.0)
    assert metrics["path1_drh0_distance_to_nearest_refuge_at_cue"] == pytest.approx(1.0)
    assert metrics["path1_drh0_refuge_reachability_success_rate"] == pytest.approx(1.0)
    assert metrics["path1_drh0_pre_consequence_refuge_arrival_rate"] == pytest.approx(1.0)


def test_path1_drh0_no_refuge_control_blocks_reachability() -> None:
    metrics = path1_drh0_generation_metrics(_env(_cfg(), refuge=None))
    assert metrics["path1_refuge_zone_cell_count"] == pytest.approx(0.0)
    assert metrics["path1_refuge_reachable_zone_cell_count"] == pytest.approx(0.0)
    assert metrics["path1_refuge_agents_reachable_refuge_at_cue"] == pytest.approx(0.0)


def test_path1_drh0_no_observation_control_does_not_change_reporting() -> None:
    off = path1_drh0_generation_metrics(_env(_cfg(observation=False)))
    on = path1_drh0_generation_metrics(_env(_cfg(observation=True)))
    assert off == on


def test_path1_drh0_no_damage_control_uses_existing_passive_wdl_proxies() -> None:
    metrics = path1_drh0_generation_metrics(
        _env(_cfg()),
        generation_metrics={
            "wdl_cue_exposed_next_move_away_count": 3.0,
            "wdl_cue_exposed_next_move_toward_count": 1.0,
            "wdl_hazard_exposure_after_cue_count": 0.0,
            "wdl_delayed_hazard_damage_total": 2.5,
        },
    )
    assert metrics["path1_refuge_move_toward_refuge_pre_active"] == pytest.approx(3.0)
    assert metrics["path1_refuge_move_away_from_refuge_pre_active"] == pytest.approx(1.0)
    assert metrics["path1_refuge_agents_stayed_in_hazard_until_active"] == pytest.approx(0.0)
    assert metrics["path1_drh0_movement_toward_refuge_after_cue"] == pytest.approx(3.0)
    assert metrics["path1_drh0_cue_conditioned_refuge_movement_delta"] == pytest.approx(2.0)
    assert metrics["path1_drh0_damage_outside_refuge"] == pytest.approx(2.5)
    assert metrics["path1_drh0_damage_prevented_by_refuge"] == pytest.approx(0.0)
