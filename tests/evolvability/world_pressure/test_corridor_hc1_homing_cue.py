from __future__ import annotations

import hashlib

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.patch_world import (
    CORRIDOR_CP0_ZONE_CORRIDOR,
    CORRIDOR_CP0_ZONE_HOME_LOCAL,
    CORRIDOR_CP0_ZONE_REMOTE_RICH,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    apply_corridor_hc1_homing_cue_overlay,
    initialize_patch_world_arrays,
)
from tools.bin.corridor_hc1_probe_construct_search import route_state_blocker, symmetry_resolution


def _cfg(*, enabled: bool) -> EnvConfig:
    cfg = EnvConfig(width=14, height=6, view_radius=1)
    cfg.patch_world_enable = True
    cfg.corridor_cp0_enable = True
    cfg.corridor_cp0_home_width = 4
    cfg.corridor_cp0_remote_width = 4
    cfg.corridor_cp0_corridor_half_width = 0
    cfg.corridor_hc1_homing_cue_enable = bool(enabled)
    cfg.corridor_hc1_homing_cue_field = "shelter"
    return cfg


def _reflected_arrays(cfg: EnvConfig) -> dict[str, np.ndarray]:
    arrays = initialize_patch_world_arrays(cfg, np.zeros((cfg.height, cfg.width), dtype=np.int8))
    for name, value in list(arrays.items()):
        arrays[name] = np.fliplr(value).copy()
    apply_corridor_hc1_homing_cue_overlay(cfg, arrays)
    return arrays


def _obs_shelter_slice(arrays: dict[str, np.ndarray], y: int, x: int) -> np.ndarray:
    shelter = np.asarray(arrays["patch_shelter"], dtype=np.float32)
    out = np.zeros((3, 3), dtype=np.float32)
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < shelter.shape[0] and 0 <= xx < shelter.shape[1]:
                out[iy, ix] = shelter[yy, xx]
    return out


def test_corridor_hc1_defaults_disabled_and_no_zone_observation() -> None:
    cfg = EnvConfig()

    assert cfg.corridor_hc1_homing_cue_enable is False
    assert cfg.corridor_hc1_homing_cue_field == "shelter"
    assert "corridor_cp0_zone_class" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES


def test_corridor_hc1_shelter_gradient_resolves_reflected_home_ambiguity() -> None:
    cfg = _cfg(enabled=True)
    normal = initialize_patch_world_arrays(cfg, np.zeros((cfg.height, cfg.width), dtype=np.int8))
    reflected = _reflected_arrays(cfg)
    y = cfg.height // 2
    normal_slice = _obs_shelter_slice(normal, y, 1)
    reflected_slice = _obs_shelter_slice(reflected, y, cfg.width - 2)

    assert not np.array_equal(normal_slice, reflected_slice)
    assert normal_slice[1, 2] < normal_slice[1, 1] < normal_slice[1, 0]
    assert reflected_slice[1, 0] < reflected_slice[1, 1] < reflected_slice[1, 2]
    assert hashlib.sha256(normal_slice.tobytes()).hexdigest() != hashlib.sha256(reflected_slice.tobytes()).hexdigest()


def test_corridor_hc1_disabled_preserves_cp0_step_shelter_values() -> None:
    cfg = _cfg(enabled=False)
    arrays = initialize_patch_world_arrays(cfg, np.zeros((cfg.height, cfg.width), dtype=np.int8))
    zones = arrays["corridor_cp0_zone_class"].astype(np.int32)

    assert arrays["patch_shelter"][zones == CORRIDOR_CP0_ZONE_HOME_LOCAL][0] == pytest.approx(cfg.corridor_cp0_home_shelter)
    assert arrays["patch_shelter"][zones == CORRIDOR_CP0_ZONE_CORRIDOR][0] == pytest.approx(cfg.corridor_cp0_corridor_shelter)
    assert arrays["patch_shelter"][zones == CORRIDOR_CP0_ZONE_REMOTE_RICH][0] == pytest.approx(cfg.corridor_cp0_remote_shelter)


def test_corridor_hc1_construct_record_solves_reflection_not_route_phase() -> None:
    symmetry = symmetry_resolution()
    blocker = route_state_blocker()

    assert symmetry["ordinary_observations_identical_after_hc1"] is False
    assert symmetry["normal_lower_shelter_neighbor_points_to_route"] is True
    assert symmetry["reflected_lower_shelter_neighbor_points_to_route"] is True
    assert blocker["ordinary_observations_identical_at_equal_energy"] is True
    assert blocker["outbound_required_action"] != blocker["inbound_required_action"]
