from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.env_config import EnvConfig
from env.gridworld import FOOD
from env.gridworld_pkg.jungle import validate_jungle_config
from env.gridworld_pkg.patch_world import initialize_patch_world_arrays, roz2a_patch_niche_zone_specs
from evolution.population_pkg.population_config import PopConfig
from simulation.branch_observability import NICHE_LIFECYCLE_FILENAME
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _active_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=8, height=8)
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.patch_movement_cost_apply_enable = True
    cfg.patch_world_waste_consequence_enable = True
    cfg.jungle_ecological_recovery_enable = True
    cfg.roz2a_patch_niche_enable = True
    cfg.roz2a_patch_niche_radius = 1
    return cfg


def _tiny_sim(run_dir: Path, *, seed: int = 123, active: bool = False, reporting: bool = False) -> SimulationRunner:
    torch.manual_seed(int(seed))
    env_cfg = _active_env_cfg() if active else EnvConfig(width=8, height=8)
    env_cfg.patch_world_enable = True
    env_cfg.patch_world_observation_enable = True
    env_cfg.reproduce_requires_mate = False
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_cooldown_steps = 0
    pop_cfg = PopConfig(population_size=8, parent_fraction=0.5, cull_fraction=0.0, slot_factor=6)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=2,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    sim_cfg.branch_observability_enable = bool(reporting)
    sim_cfg.branch_niche_observability_enable = bool(reporting)
    sim_cfg.branch_observability_run_id = "roz2a_unit"
    sim_cfg.lineage_reproduction_funnel_reporting_enable = bool(reporting)
    if reporting:
        sim_cfg.lineage_reproduction_funnel_write_events = True
        sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
        sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
        sim_cfg.path2_roz_reporting_enable = True
        sim_cfg.path2_roz_schedule_mode = "patch_derived_static"
        sim_cfg.path2_roz_schedule_id = "patch_static_unit"
        sim_cfg.path2_roz_window_period_steps = 1
        sim_cfg.path2_roz_window_duration_steps = 1
        sim_cfg.path2_roz_zone_radius = 1
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(run_dir), perf=False)


def _fingerprint(sim: SimulationRunner) -> str:
    payload = []
    for agent in sorted(sim.pop.agents.values(), key=lambda item: int(item.agent_id)):
        payload.append(
            {
                "agent_id": int(agent.agent_id),
                "alive": bool(agent.alive),
                "energy": round(float(agent.energy), 8),
                "health": round(float(agent.health), 8),
                "lineage_id": str(agent.stats.lineage_id),
                "offspring_count": int(agent.stats.offspring_count),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _zone_center(specs: list[dict], zone_id: str) -> tuple[int, int]:
    spec = next(item for item in specs if item["id"] == zone_id)
    return int(spec["center_y"]), int(spec["center_x"])


def test_roz2a_patch_niche_defaults_off() -> None:
    cfg = EnvConfig()

    assert cfg.roz2a_patch_niche_enable is False
    assert cfg.roz2a_patch_niche_mode == "static_three_zone_tradeoff_v1"


def test_roz2a_patch_niche_requires_patch_world() -> None:
    cfg = EnvConfig(width=8, height=8)
    cfg.roz2a_patch_niche_enable = True

    with pytest.raises(ValueError, match="roz2a_patch_niche_enable requires patch_world_enable"):
        validate_jungle_config(cfg)


def test_roz2a_patch_niche_static_overlay_creates_tradeoff_arrays() -> None:
    cfg = _active_env_cfg()
    grid = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    specs = roz2a_patch_niche_zone_specs(cfg, shape=grid.shape)
    for zone_id in ("resource_exposed", "sheltered_low_resource", "transition_mixed"):
        yy, xx = _zone_center(specs, zone_id)
        grid[yy, xx] = FOOD

    arrays = initialize_patch_world_arrays(cfg, grid)
    ry, rx = _zone_center(specs, "resource_exposed")
    sy, sx = _zone_center(specs, "sheltered_low_resource")
    ty, tx = _zone_center(specs, "transition_mixed")

    assert arrays["patch_fertility"][ry, rx] == pytest.approx(cfg.roz2a_patch_niche_resource_fertility)
    assert arrays["patch_movement_cost"][ry, rx] == pytest.approx(cfg.roz2a_patch_niche_resource_movement_cost)
    assert arrays["patch_waste"][ry, rx] == pytest.approx(cfg.roz2a_patch_niche_resource_waste)
    assert arrays["patch_food_amount"][sy, sx] == pytest.approx(cfg.roz2a_patch_niche_shelter_food_multiplier)
    assert arrays["patch_cover"][sy, sx] == pytest.approx(cfg.roz2a_patch_niche_shelter_cover)
    assert arrays["patch_shelter"][sy, sx] == pytest.approx(cfg.roz2a_patch_niche_shelter_shelter)
    assert arrays["patch_recovery_activity"][sy, sx] == pytest.approx(cfg.roz2a_patch_niche_shelter_recovery)
    assert arrays["patch_fertility"][ty, tx] == pytest.approx(cfg.roz2a_patch_niche_transition_fertility)
    assert arrays["patch_cover"][ty, tx] == pytest.approx(cfg.roz2a_patch_niche_transition_cover)


def test_roz2a_reporting_only_preserves_trajectory_but_active_is_deterministic(tmp_path: Path) -> None:
    baseline = _tiny_sim(tmp_path / "baseline", seed=444, active=False, reporting=False)
    baseline.run(start_gen=0, skip_init=False)
    reporting = _tiny_sim(tmp_path / "reporting", seed=444, active=False, reporting=True)
    reporting.run(start_gen=0, skip_init=False)
    active_a = _tiny_sim(tmp_path / "active_a", seed=444, active=True, reporting=True)
    active_a.run(start_gen=0, skip_init=False)
    active_b = _tiny_sim(tmp_path / "active_b", seed=444, active=True, reporting=True)
    active_b.run(start_gen=0, skip_init=False)

    assert _fingerprint(reporting) == _fingerprint(baseline)
    assert _fingerprint(active_a) == _fingerprint(active_b)
    assert (tmp_path / "active_a" / "branch_observability" / NICHE_LIFECYCLE_FILENAME).read_bytes() == (
        tmp_path / "active_b" / "branch_observability" / NICHE_LIFECYCLE_FILENAME
    ).read_bytes()


def test_roz2a_patch_niche_checkpoint_restores_patch_arrays(tmp_path: Path) -> None:
    sim = _tiny_sim(tmp_path / "part", seed=515, active=True, reporting=False)
    before = {
        name: np.asarray(getattr(sim.env, name), dtype=np.float32).copy()
        for name in ("patch_food_amount", "patch_fertility", "patch_cover", "patch_shelter", "patch_movement_cost", "patch_waste")
    }
    bundle = checkpoint_bundle_v2_save(sim, str(tmp_path / "bundle_gen0"), generation_index=0, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    resumed = _tiny_sim(tmp_path / "resume", seed=515, active=True, reporting=False)
    for name in before:
        getattr(resumed.env, name).fill(0.0)
    checkpoint_bundle_v2_apply(resumed, payload, restore_rng=True)

    for name, expected in before.items():
        assert np.array_equal(np.asarray(getattr(resumed.env, name), dtype=np.float32), expected)
